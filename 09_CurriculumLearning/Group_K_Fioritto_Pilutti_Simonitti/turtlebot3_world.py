import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import math
import random
from collections import deque, defaultdict
from tf.transformations import euler_from_quaternion


class GoalSamplingCurriculum:
    def __init__(self, all_goals, epsilon=0.2, achieved_maxlen=300, perf_window=20):
        self.all_goals = list(all_goals)
        self.epsilon = float(epsilon)
        self.achieved_goals = deque(maxlen=achieved_maxlen)
        self.perf_window = int(perf_window)
        self.goal_perf = defaultdict(lambda: deque(maxlen=self.perf_window)) # (success, steps)

    def record_episode(self, goal_xy, success, steps):
        g = tuple(goal_xy)
        self.goal_perf[g].append((int(success), int(steps)))
        if success:
            self.achieved_goals.append(g)

    def _difficulty(self, goal_xy, ref_xy=(-2.0, -0.5)):
        gx, gy = goal_xy
        rx, ry = ref_xy
        return math.hypot(gx - rx, gy - ry)

    def _challenging_score(self, goal_xy):
        g = tuple(goal_xy)
        hist = self.goal_perf[g]
        if len(hist) == 0:
            return 0.0
        sr = sum(s for s, _ in hist) / float(len(hist)) # success rate
        avg_steps = sum(st for _, st in hist) / float(len(hist)) # average steps
        # Prefer medium success rates (not super easy or super hard)
        mid = 1.0 - abs(sr - 0.55) / 0.55
        mid = max(0.0, mid)
        return mid * (1.0 + avg_steps / 200.0)

    def sample_goal(self, ref_xy=(-2.0, -0.5)):
        # Explore
        if random.random() < self.epsilon or len(self.achieved_goals) == 0:
            diffs = [self._difficulty(g, ref_xy) for g in self.all_goals]
            weights = [(d + 1e-6) ** 1.5 for d in diffs] # bias to farther goals
            return random.choices(self.all_goals, weights=weights, k=1)[0]

        # Replay challenging achieved goals
        candidates = list(set(self.achieved_goals))
        scores = [self._challenging_score(g) for g in candidates]
        if max(scores) <= 1e-9:
            return random.choice(candidates)
        return random.choices(candidates, weights=scores, k=1)[0]



class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        high = numpy.full((num_laser_readings), self.max_laser_value)
        low = numpy.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        self.cumulated_steps = 0.0

        # Define the set of goal points (discrete goals) for goal sampling
        self.goal_threshold = rospy.get_param("/turtlebot3/goal_threshold", 0.25)
        self.max_steps_per_episode = rospy.get_param("/turtlebot3/max_steps_per_episode", 600)

        
        # Parametri di Ponderazione (Weights)
        self.w_progress = rospy.get_param("/turtlebot3/progress_rwd", 40.0)
        self.w_collision = rospy.get_param("/turtlebot3/collision_rwd", 2.0)
        self.w_smooth = rospy.get_param("/turtlebot3/smooth_rwd", 0.15)

        # Valori Terminali
        self.terminal_goal = rospy.get_param("/turtlebot3/terminal_goal_rwd", 50.0)
        self.terminal_crash = rospy.get_param("/turtlebot3/terminal_crash_rwd", -25.0)
        self.terminal_timeout = rospy.get_param("/turtlebot3/terminal_timeout_rwd", -10.0)

        # Penalità temporale costante per ogni step
        self.r_time = rospy.get_param("/turtlebot3/time_rwd", -0.05)

        #self.easy_goals = [
        #    (-1.6, -0.5),   # +x
        #    (-2.4, -0.5),   # -x
        #    (-2.0, -0.1),   # +y
        #    (-2.0, -0.9),   # -y
        #
        #    (-1.4, -0.5),
        #    (-2.6, -0.5),
        #    (-2.0,  0.1),
        #    (-2.0, -1.1),
        #
        #    (-1.2, -0.5),
        #    (-2.8, -0.5),
        #    (-2.0,  0.3),
        #    (-2.0, -1.3),
        #]
        #self.mid_goals = [
        #    # cardinali
        #    (-1.0, -0.5),
        #    (-3.0, -0.5),
        #    (-2.0,  0.5),
        #    (-2.0, -1.5),
        #
        #    (-0.8, -0.5),
        #    (-3.2, -0.5),
        #    (-2.0,  0.7),
        #    (-2.0, -1.7),
        #
        #    # diagonali moderate
        #    (-1.3, -0.2),
        #    (-1.3, -0.8),
        #    (-2.7, -0.2),
        #    (-2.7, -0.8),
        #
        #    (-1.1,  0.1),
        #    (-1.1, -1.1),
        #    (-2.9,  0.1),
        #    (-2.9, -1.1),
        #]
        #self.hard_goals = [
        #    # cardinali lontani
        #    (-0.2, -0.5),
        #    (-3.8, -0.5),
        #    (-2.0,  1.5),
        #    (-2.0, -2.5),
        #
        #    # diagonali lunghe
        #    (-0.6,  0.3),
        #    (-0.6, -1.3),
        #    (-3.4,  0.3),
        #    (-3.4, -1.3),
        #
        #    (-0.2,  0.9),
        #    (-0.2, -1.9),
        #    (-3.8,  0.9),
        #    (-3.8, -1.9),
        #]

        self.easy_goals = [(-1.0, -0.5), (-1.5, 0.5), (-2.0, -1.5)]
        self.mid_goals  = [(0.0, 0.0), (-0.5, -1.5), (0.5, 1.0)]
        self.hard_goals = [(2.0, 2.0), (1.5, -2.0), (2.0, 0.0)]

        self.discrete_goals = (
            self.easy_goals +
            self.mid_goals +
            self.hard_goals
        )

        self.goal_xy = self.discrete_goals[0]  # Initialize goal

        self.curriculum = GoalSamplingCurriculum(all_goals=self.discrete_goals, epsilon=0.2)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True

    # Use odom ros topic to get the robot pose: xy coordinates and yaw
    def _get_robot_pose(self):
        odom = self.get_odom()
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return float(p.x), float(p.y), float(yaw)

    # Calculate distance from robot to goal
    def _distance_to_goal(self, x, y):
        gx, gy = self.goal_xy
        return math.hypot(gx - x, gy - y)

    def _sample_goal(self):
    # Calcola il success rate globale o recente (es. ultimi 50 ep)
    # Nota: Questo richiede di salvare lo storico successi fuori da GoalSamplingCurriculum o interrogarlo meglio.
    # Assumiamo di poter accedere a self.curriculum.achieved_goals

        total_achieved = len(self.curriculum.achieved_goals)

        # Logica a Fasi (Stages)
        # Stage 1: Solo easy goals finché non ne padroneggiamo un po'
        if total_achieved < 15: 
            self.goal_xy = random.choice(self.easy_goals)
            self.goal_sample_mode = "easy_training"

        # Stage 2: Mischia Easy e Mid
        elif total_achieved < 40:
            if random.random() < 0.5:
                self.goal_xy = random.choice(self.easy_goals)
            else:
                self.goal_xy = random.choice(self.mid_goals)
            self.goal_sample_mode = "mid_training"

        # Stage 3: Curriculum completo (usa la tua classe intelligente)
        else:
            # Usa la logica intelligente definita nella classe GoalSamplingCurriculum
            self.goal_xy = self.curriculum.sample_goal(ref_xy=self._get_robot_pose()[:2])
            self.goal_sample_mode = "curriculum_smart"


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0

        # New buffers for the cumulative components of the reward
        self.cum_r_progress = 0.0
        self.cum_r_time = 0.0
        self.cum_r_smooth = 0.0
        self.cum_r_collision_avoid = 0.0
        self.cum_r_terminal = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # Episode termination includes goal success + timeout
        self.steps = 0
        self.reached_goal = False
        self.collision = False
        self.prev_dist = None
        self.prev_action = None
        self._episode_logged = False
        self._sample_goal()
        rospy.logerr(f"[RESET] New goal sampled: {self.goal_xy}")
        rospy.logerr(f"[RESET] Total achieved goals so far: {len(self.curriculum.achieved_goals)}")
        rospy.logerr(f"[RESET] Goal sampling mode: {self.goal_sample_mode}")


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        self.last_action = int(action)

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations

    def _on_episode_end(self):
        self.curriculum.record_episode(self.goal_xy, self.reached_goal, self.steps)


    def _is_done(self, observations):
        """
        Episode termination conditions:
        1) Goal reached (success)
        2) Timeout (max steps)
        3) Collision / unsafe proximity (laser logic sets self._episode_done)
        4) Crash (IMU acceleration spike)
        Also:
        - sets self.collision when termination is not goal and not timeout
        - triggers _on_episode_end exactly once
        """

        # Default flags (in case something forgot to init them)
        if not hasattr(self, "_episode_done"):
            self._episode_done = False
        if not hasattr(self, "reached_goal"):
            self.reached_goal = False
        if not hasattr(self, "collision"):
            self.collision = False
        if not hasattr(self, "_episode_logged"):
            self._episode_logged = False

        # --- Compute distance to goal ---
        x, y, _ = self._get_robot_pose()
        dist = self._distance_to_goal(x, y)
        rospy.logwarn(f"Distance to goal: {dist:.3f}")

        # --- 1) Success condition: reached goal ---
        if dist <= self.goal_threshold:
            self.reached_goal = True
            self._episode_done = True
            rospy.loginfo("Reached goal (SUCCESS).")

        # --- 2) Timeout condition ---
        if (not self._episode_done) and (self.steps >= self.max_steps_per_episode):
            self._episode_done = True
            rospy.loginfo("Reached max steps (TIMEOUT).")

        # --- 3) Laser-based collision/unsafe proximity ---
        vals = [float(v) for v in observations if v is not None and math.isfinite(float(v)) and float(v) > 1e-6]
        min_scan = min(vals) if len(vals) > 0 else float(self.max_laser_value)
        if (not self._episode_done) and (min_scan < self.min_range):
            self._episode_done = True
            rospy.logwarn(f"Laser collision detected: min_scan={min_scan:.3f} < min_range={self.min_range:.3f}")
        else:
            # This might be goal/timeout too, so keep it as debug-ish
            rospy.logdebug("Episode marked done (could be goal/timeout/laser).")

        # --- 4) IMU crash detection (only if not already success/timeout) ---
        # You can still keep this even if episode already done, but it’s cleaner to avoid overrides.
        if not self.reached_goal and (self.steps < self.max_steps_per_episode):
            imu_data = self.get_imu()
            lin_acc_mag = self.get_vector_magnitude(imu_data.linear_acceleration)

            if lin_acc_mag > self.max_linear_aceleration:
                self._episode_done = True
                rospy.logwarn(
                    f"IMU crash detected: {lin_acc_mag:.3f} > {self.max_linear_aceleration:.3f}"
                )

        # --- Mark collision cause (only if done but not success and not timeout) ---
        if self._episode_done and (not self.reached_goal) and (self.steps < self.max_steps_per_episode):
            self.collision = True

        # --- Episode end hook (exactly once) ---
        if self._episode_done and not self._episode_logged:
            self._episode_logged = True
            try:
                self._on_episode_end()
            except Exception as e:
                rospy.logerr(f"_on_episode_end() failed: {e}")

        rospy.logwarn(f"Achieved Goals: {self.curriculum.achieved_goals}")

        return self._episode_done

    def _compute_reward(self, observations, done):
        # --- 1. State and Distance ---
        x, y, _ = self._get_robot_pose()
        dist = self._distance_to_goal(x, y)

        if self.prev_dist is None:
            self.prev_dist = dist

        # --- 2. Compute factorized components ---

        # A. Progress Reward: based on distance difference (Reward Shaping)
        # If diff > 0 the robot got closer, if < 0 it got farther
        diff = self.prev_dist - dist
        r_progress = self.w_progress * diff

        # B. Safety / Collision Avoidance
        # Compute min valid laser scan
        valid_scan = [o for o in observations if not numpy.isinf(o) and not numpy.isnan(o)]
        min_scan = min(valid_scan) if len(valid_scan) > 0 else 10.0

        r_collision_avoid = 0.0
        threshold_safe = 0.3 # meters
        if min_scan < threshold_safe:
            # Linear penalty: the closer it is, the more negative the reward
            r_collision_avoid = -self.w_collision * (threshold_safe - min_scan)
        else:
            r_collision_avoid = 0.1 * self.w_collision  # Small reward for being in a safe zone

        # C. Smooth Steering
        # Penalize if last action was not FORWARDS
        r_smooth = 0.0
        if self.last_action != "FORWARDS": 
            r_smooth = -self.w_smooth

        # D. Terminal Rewards
        r_terminal = 0.0
        if done:
            if self.reached_goal:
                r_terminal = self.terminal_goal
            elif self.steps >= self.max_steps_per_episode:
                r_terminal = self.terminal_timeout
            else:
                # If done is True but not goal or timeout, it's a collision
                r_terminal = self.terminal_crash

        # E. Time Rewards
        r_cumm_time = self.r_time * self.steps

        # --- 4. Final computation and update ---
        reward = r_progress + r_cumm_time  + r_smooth + r_collision_avoid + r_terminal

        # Update cumulative reward components
        self.cum_r_progress += r_progress
        self.cum_r_time += self.r_time
        self.cum_r_smooth += r_smooth
        self.cum_r_collision_avoid += r_collision_avoid
        self.cum_r_terminal += r_terminal

        # Update state variables for next step
        self.prev_dist = dist
        self.prev_action = self.last_action

        # --- 5. Logging structure ---
        self.last_reward_components = {
            "r_progress": float(r_progress),
            "r_time": float(self.r_time),
            "r_smooth": float(r_smooth),
            "r_collision_avoid": float(r_collision_avoid),
            "r_terminal": float(r_terminal),
            "dist": float(dist),
            "min_scan": float(min_scan),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "r_total": float(reward),
            "cum_r_progress": self.cum_r_progress,
            "cum_r_time": self.cum_r_time,
            "cum_r_smooth": self.cum_r_smooth,
            "cum_r_collision_avoid": self.cum_r_collision_avoid,
            "cum_r_terminal": self.cum_r_terminal,
            "cumulated_reward": self.cumulated_reward,
        }

        self.cumulated_reward += reward
        self.steps += 1

        print("Step Reward: {:.2f} | Components: {}".format(
            reward, self.last_reward_components))

        return reward




        # Internal TaskEnv Methods

        #def discretize_scan_observation(self,data,new_ranges):
        """
        #Discards all the laser readings that are not multiple in index of new_ranges
        #value.
        #"""
        #self._episode_done = False

        #discretized_ranges = []
        #mod = len(data.ranges)/new_ranges

        #rospy.logdebug("data=" + str(data))
        #rospy.logdebug("new_ranges=" + str(new_ranges))
        #rospy.logdebug("mod=" + str(mod))

        #for i, item in enumerate(data.ranges):
        #    if (i%mod==0):
        #        if item == float ('Inf') or numpy.isinf(item):
        #            discretized_ranges.append(self.max_laser_value)
        #        elif numpy.isnan(item):
        #            discretized_ranges.append(self.min_laser_value)
        #        else:
        #            discretized_ranges.append(item)

        #        if (self.min_range > item > 0):
        #            rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
        #            self._episode_done = True
        #        else:
        #            rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        #return discretized_ranges

    def discretize_scan_observation(self, data, new_ranges):
        """
        Downsample laser scan to new_ranges beams and set self._episode_done if too close.
        IMPORTANT: keep floats (do NOT int-cast), ignore invalid values.
        """
        self._episode_done = False
        discretized_ranges = []

        n = len(data.ranges)
        step = max(1, int(n / float(new_ranges)))  # integer stride

        for i in range(0, n, step):
            item = data.ranges[i]

            # handle invalid / inf / nan
            if item == float('Inf') or numpy.isinf(item):
                val = float(self.max_laser_value)
            elif numpy.isnan(item):
                val = float(self.max_laser_value)  # <-- IMPORTANT: do NOT set to min
            else:
                val = float(item)

            # keep float (optionally round to reduce state space)
            discretized_ranges.append(round(val, 2))

        return discretized_ranges
        


    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

