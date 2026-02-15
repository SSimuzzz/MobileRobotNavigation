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
from collections import deque, defaultdict, Counter
from tf.transformations import euler_from_quaternion


class GoalSamplingCurriculum:
    def __init__(self, all_goals, final_goal, epsilon=0.2, achieved_maxlen=10000, perf_window=20, easy_goal_thold=0.75):
        self.all_goals = list(all_goals)
        self.epsilon = float(epsilon)
        self.achieved_goals = deque(maxlen=achieved_maxlen)
        self.final_goal = final_goal
        self.perf_window = int(perf_window)
        self.easy_goal_thold = float(easy_goal_thold)
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

        if sr > self.easy_goal_thold:
            return 0.0

        avg_steps = sum(st for _, st in hist) / float(len(hist)) # average steps
        # Prefer medium success rates (not super easy or super hard)
        mid = 1.0 - abs(sr - 0.55) / 0.55
        mid = max(0.0, mid)
        return mid * (1.0 + avg_steps / 200.0)

    """def sample_goal(self, ref_xy=(-2.0, -0.5)):
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
        return random.choices(candidates, weights=scores, k=1)[0]"""

    def sample_goal(self, ref_xy):
        max_per_goal = 100

        if len(self.achieved_goals) > len(set(self.achieved_goals)) * max_per_goal:
            return self.final_goal

        counter = Counter(self.achieved_goals)

        valid_goals = [goal for goal, count in counter.items() if count <= max_per_goal]


        if random.random() < self.epsilon or not valid_goals:
            return random.choice(self.all_goals)
        
        return random.choice(list(set(valid_goals)))


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
        
        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")
                               
        # -----------------------------
        # Training phase
        # -----------------------------
        self.training_phase = rospy.get_param("/turtlebot3/training_phase", 1)
        print(f"[TRAINING PHASE] {self.training_phase}")
        
        if self.training_phase == 1:
            launch_file_name = "start_world_ph1.launch"
        elif self.training_phase == 2:
            launch_file_name = "start_world_ph2.launch"
        elif self.training_phase == 3:
            launch_file_name = "start_world.launch"
        else:
            launch_file_name = "start_world.launch"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name=launch_file_name,
                    ros_ws_abspath=ros_ws_abspath)


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

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


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        high = numpy.full((num_laser_readings), self.max_laser_value)
        low = numpy.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        #self.observation_space = spaces.Box(low, high)

        # ---- Extra observation bounds: distance (raw) + heading normalized ----
        # distance in meters: [0, max_goal_distance]
        self.max_goal_distance = rospy.get_param("/turtlebot3/max_goal_distance", 10.0)
        self.progress_clip = rospy.get_param("/turtlebot3/progress_clip", 10.0)
        self.collision_clip = rospy.get_param("/turtlebot3/collision_clip", 15.0)

        # Aggiungo 2 feature: dist_norm in [0, 1] (se dividi per 10 e clampi) e heading_norm in [-1, 1]
        low_extra = numpy.array([0.0, -1.0], dtype=numpy.float32)
        high_extra = numpy.array([1.0,  1.0], dtype=numpy.float32)

        low_obs = numpy.concatenate([low.astype(numpy.float32), low_extra])
        high_obs = numpy.concatenate([high.astype(numpy.float32), high_extra])

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=numpy.float32)


        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        easy_goal_thold = rospy.get_param("/turtlebot3/easy_goal_thold")

        self.cumulated_steps = 0.0

        # Define the set of goal points (discrete goals) for goal sampling
        self.goal_threshold = rospy.get_param("/turtlebot3/goal_threshold", 0.25)
        self.max_steps_per_episode = rospy.get_param("/turtlebot3/max_steps_per_episode", 600)

        
        # Parametri di Ponderazione (Weights)
        self.w_progress = rospy.get_param("/turtlebot3/progress_rwd", 40.0)

        self.w_collision_ph1 = rospy.get_param("/turtlebot3/collision_rwd_ph1", 2.0)
        self.w_collision_ph2 = rospy.get_param("/turtlebot3/collision_rwd_ph2", 2.0)
        self.w_collision_ph3 = rospy.get_param("/turtlebot3/collision_rwd_ph3", 2.0)
        self.w_collision_ph4 = rospy.get_param("/turtlebot3/collision_rwd_ph4", 2.0)
        
        self.w_yaw_ph1 = rospy.get_param("/turtlebot3/yaw_rwd_ph1", 1.0)
        self.w_yaw_ph2 = rospy.get_param("/turtlebot3/yaw_rwd_ph2", 1.0)
        self.w_yaw_ph3 = rospy.get_param("/turtlebot3/yaw_rwd_ph3", 1.0)
        self.w_yaw_ph4 = rospy.get_param("/turtlebot3/yaw_rwd_ph4", 1.0)

        # Valori Terminali
        self.terminal_goal = rospy.get_param("/turtlebot3/terminal_goal_rwd", 50.0)
        self.terminal_crash = rospy.get_param("/turtlebot3/terminal_crash_rwd", -25.0)
        self.terminal_timeout = rospy.get_param("/turtlebot3/terminal_timeout_rwd", -10.0)

        # Penalità temporale costante per ogni step
        self.r_time = rospy.get_param("/turtlebot3/time_rwd", -0.05)

        self.easy_goals = [(-0.7, -0.5)]
        self.mid_goals  = [(0.0, 0.5)]
        self.hard_goals = [(1.0, -0.5)]
        self.final_goal = [(1.7, 0.5)]

        self.discrete_goals = (
            self.easy_goals +
            self.mid_goals +
            self.hard_goals +
            self.final_goal
        )

        self.curriculum = GoalSamplingCurriculum(all_goals=self.discrete_goals, final_goal = self.final_goal[0], epsilon=0.2, easy_goal_thold=easy_goal_thold)

        self._sample_goal() # Initialize goal

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
    
    def _normalize_angle(self, a):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _heading_to_goal(self, x, y, yaw):
        """Return heading error to goal in [-pi, pi]."""
        gx, gy = self.goal_xy
        goal_bearing = math.atan2(gy - y, gx - x)   # angolo assoluto verso il goal (world frame)
        heading_error = self._normalize_angle(goal_bearing - yaw)
        return heading_error


    #def _sample_goal(self):
    ## Calcola il success rate globale o recente (es. ultimi 50 ep)
    ## Nota: Questo richiede di salvare lo storico successi fuori da GoalSamplingCurriculum o interrogarlo meglio.
    ## Assumiamo di poter accedere a self.curriculum.achieved_goals
#
    #    total_achieved = len(self.curriculum.achieved_goals)
#
    #    # Logica a Fasi (Stages)
    #    # Stage 1: Solo easy goals finché non ne padroneggiamo un po'
    #    if total_achieved < 15: 
    #        self.goal_xy = random.choice(self.easy_goals)
    #        self.goal_sample_mode = "easy_training"
#
    #    # Stage 2: Mischia Easy e Mid
    #    elif total_achieved < 40:
    #        if random.random() < 0.5:
    #            self.goal_xy = random.choice(self.easy_goals)
    #        else:
    #            self.goal_xy = random.choice(self.mid_goals)
    #        self.goal_sample_mode = "mid_training"
#
    #    # Stage 3: Curriculum completo (usa la tua classe intelligente)
    #    else:
    #        # Usa la logica intelligente definita nella classe GoalSamplingCurriculum
    #        self.goal_xy = self.curriculum.sample_goal(ref_xy=self._get_robot_pose()[:2])
    #        self.goal_sample_mode = "curriculum_smart"
        
    # ============================================================
    #  Goal sampling (PHASE-AWARE)
    # ============================================================
    def _sample_goal(self):
        if self.training_phase == 1:
            self.goal_xy = random.choice(self.discrete_goals)
            self.goal_sample_mode = "random goals"
        elif self.training_phase == 2:
            self.goal_xy = self.hard_goals[0]
            self.goal_sample_mode = "fixed easy goal"
        elif self.training_phase == 3:
            self.goal_xy = random.choice(self.mid_goals + self.hard_goals)
            self.goal_sample_mode = "fixed hard goal"
        else:
            self.goal_sample_mode = "curriculum smart"
            self.goal_xy = self.curriculum.sample_goal(self._get_robot_pose()[:2])

        rospy.logwarn(f"[GOAL] {self.goal_xy}")


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

        self.goal_angle = 0.0

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
        self.scan_sub_callback(discretized_observations)

        # normalize between 0 and 1
        discretized_observations = [
            min(x / self.max_laser_value, 1.0)
            for x in discretized_observations
        ]


        # 2) Goal features (distance + heading error)
        x, y, yaw = self._get_robot_pose()
        dist = self._distance_to_goal(x, y)
        heading = self._heading_to_goal(x, y, yaw)
        self.goal_angle = heading
        self.goal_dist = dist

        dist_norm = dist / (self.max_goal_distance + 1e-8)

        sin_heading = math.sin(heading)
        cos_heading = math.cos(heading)

        #heading_norm = heading / math.pi     # now is in [-1, 1]

        #obs = list(discretized_observations) + [dist_norm, heading_norm]
        obs = list(discretized_observations) + [dist_norm, sin_heading, cos_heading]

        print("Observations==>" + str(obs))
        rospy.logdebug("Observations==>" + str(obs))
        rospy.logdebug("END Get Observation ==>")
        return obs

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
        vals = [float(v) for v in observations[:self.new_ranges] 
                if v is not None 
                and math.isfinite(float(v)) 
                and float(v) > 1e-6]
        min_scan = min(vals) if len(vals) > 0 else float(self.max_laser_value)
        min_scan = min_scan * self.max_laser_value # de-normalize
        if (not self._episode_done) and (min_scan < self.min_range):
            self._episode_done = True
            rospy.logwarn(f"Laser collision detected: min_scan={min_scan:.3f} < min_range={self.min_range:.3f}")
        else:
            # This might be goal/timeout too, so keep it as debug-ish
            rospy.logdebug("Episode marked done (could be goal/timeout/laser).")


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
    
    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 6
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights

    def compute_weighted_obstacle_reward(self):
        if not self.front_ranges or not self.front_angles:
            return 0.0

        front_ranges = numpy.array(self.front_ranges)
        front_angles = numpy.array(self.front_angles)

        valid_mask = front_ranges <= 0.5
        if not numpy.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = numpy.unwrap(front_angles)
        relative_angles[relative_angles > numpy.pi] -= 2 * numpy.pi

        weights = self.compute_directional_weights(relative_angles, max_weight=10.0)

        safe_dists = numpy.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = numpy.exp(-3.0 * safe_dists)

        weighted_decay = numpy.dot(weights, decay)

        reward = - (1.0 + 4.0 * weighted_decay)

        return reward

    def _compute_reward(self, observations, done):
        # --- 1. Stato e Distanze ---
        #x, y, _ = self._get_robot_pose()
        #dist = self._distance_to_goal(x, y)

        dist = self.goal_dist

        if self.prev_dist is None:
            self.prev_dist = dist

        # --- 2. Calcolo Componenti Fattorizzate ---

        # A. Progress Reward: basata sulla differenza di distanza (Reward Shaping)
        # Se diff > 0 il robot si è avvicinato, se < 0 si è allontanato
        diff = self.prev_dist - dist
        #r_progress = self.w_progress * diff
        r_progress = numpy.clip(self.w_progress * diff, -self.progress_clip, self.progress_clip)

        # B. Safety / Collision Avoidance
        # Calcoliamo la distanza minima dagli ostacoli (min_scan)
        valid_scan = [
            o for o in observations[:self.new_ranges]
            if not numpy.isinf(o)
            and not numpy.isnan(o)
        ]
        min_scan = min(valid_scan) if len(valid_scan) > 0 else 10.0

        collision_raw = self.compute_weighted_obstacle_reward()
        if self.training_phase == 1:
            r_collision_avoid = numpy.clip(self.w_collision_ph1 * collision_raw, -self.collision_clip, self.collision_clip)
        elif self.training_phase == 2:
            r_collision_avoid = numpy.clip(self.w_collision_ph2 * collision_raw, -self.collision_clip, self.collision_clip)
        elif self.training_phase == 3:
            r_collision_avoid = numpy.clip(self.w_collision_ph3 * collision_raw, -self.collision_clip, self.collision_clip)
        else:
            r_collision_avoid = numpy.clip(self.w_collision_ph4 * collision_raw, -self.collision_clip, self.collision_clip)

        # C. Yaw reward
        yaw_raw = math.cos(self.goal_angle)     # cos(goal_angle) gives +1 facing goal, -1 facing away
        #yaw_raw  = (1 - (2 * (abs(self.goal_angle) / math.pi)))

        if self.training_phase == 1:
            yaw_reward = self.w_yaw_ph1 * yaw_raw
        elif self.training_phase == 2:
            yaw_reward = self.w_yaw_ph2 * yaw_raw
        elif self.training_phase == 3:
            yaw_reward = self.w_yaw_ph3 * yaw_raw
        else:
            yaw_reward = self.w_yaw_ph4 * yaw_raw

        # D. Terminal Rewards
        r_terminal = 0.0
        if done:
            if self.reached_goal:
                r_terminal = self.terminal_goal
            elif self.steps >= self.max_steps_per_episode:
                r_terminal = self.terminal_timeout
            else:
                # Se done è True ma non è goal o timeout, è una collisione
                r_terminal = self.terminal_crash

        # E. Time Rewards
        #r_cumm_time = self.r_time * self.steps
        r_cumm_time = self.r_time

        # --- 4. Calcolo Totale e Aggiornamento ---
        reward = r_progress + r_cumm_time + yaw_reward + r_collision_avoid + r_terminal

        # Aggiornamento dei cumulativi dell'episodio
        self.cum_r_progress += r_progress
        self.cum_r_time += self.r_time
        self.cum_r_collision_avoid += r_collision_avoid
        self.cum_r_terminal += r_terminal

        # Aggiornamento variabili di stato per lo step successivo
        self.prev_dist = dist
        self.prev_action = self.last_action

        # --- 5. Struttura Logging Richiesta ---
        self.last_reward_components = {
            "r_progress": float(r_progress),
            "r_time": float(self.r_time),
            "r_smooth": 0,
            "r_collision_avoid": float(r_collision_avoid),
            "r_terminal": float(r_terminal),
            "dist": float(dist),
            "yaw_reward": float(yaw_reward),
            "min_scan": float(min_scan),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "r_total": float(reward),
            "cum_r_progress": self.cum_r_progress,
            "cum_r_time": self.cum_r_time,
            "cum_r_smooth": 0,
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

            if val < 0.0:
                val = self.min_laser_value

            # keep float (optionally round to reduce state space)
            discretized_ranges.append(round(val, 2))

        return discretized_ranges
    
    def scan_sub_callback(self, discretized_ranges):
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(discretized_ranges)
        angle_increment = int(360/num_of_lidar_rays)

        self.front_distance = discretized_ranges[0]

        for i in range(num_of_lidar_rays):
            angle = numpy.deg2rad(i * angle_increment)
            distance = discretized_ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)


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

