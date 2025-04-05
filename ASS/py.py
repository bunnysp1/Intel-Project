
import pygame
import math
import random
import sys
import time
import numpy as np

CONFIG = {
    'display': {
        'screen_width': 1100,
        'screen_height': 750,
        'fps': 60,
        'background_color': (240, 240, 240),
        'text_color': (10, 10, 10),
        'error_color': (200, 0, 0),
        'info_panel_width': 250,
    },
    'simulation': {
        'num_objects_to_spawn': 20,
        'initial_spawn_buffer': 40,
        'respawn_count': 5,
    },
    'robot': {
        'start_x': None,
        'start_y_offset': 70,
        'speed': 200,
        'pickup_range': 25,
        'drop_range': 30,
        'color': (50, 50, 200),
        'radius': 15,
    },
    'objects': {
        'target_colors': ["red", "blue", "green", "yellow", "purple", "orange","maron"],
        'other_bin_color_name': "other",
        'spawnable_colors': [ "aqua","black"],
        'color_map': {
            "red": (255, 50, 50), "blue": (50, 50, 255), "green": (50, 200, 50),
            "yellow": (255, 255, 50), "purple": (150, 50, 150), "orange": (255, 165, 0),
            "gray": (150, 150, 150),
            "other": (200, 200, 200),
            "unknown": (100, 100, 100)
        },
        'types': ['block', 'ball'],
        'min_weight': 2.00,
        'max_weight': 3.15,
        'min_roughness': 0.05,
        'max_roughness': 1.00,
        'min_size': 10,
        'max_size': 16,
    },
    'bins': {
        'max_per_row': 4,
        'width': 100,
        'height': 70,
        'h_spacing_factor': 1.0,
        'v_spacing': 30,
        'top_margin': 30,
        'highlight_factor': 50,
    },
    'ai': {
        'learning_rate': 0.15,
        'decay_rate': 0.003,
        'min_preference': 0.05,
        'default_preference': 0.5,
    }
}

CONFIG['display']['game_area_width'] = CONFIG['display']['screen_width'] - CONFIG['display']['info_panel_width']
CONFIG['robot']['start_x'] = CONFIG['display']['game_area_width'] // 2
CONFIG['robot']['start_y'] = CONFIG['display']['screen_height'] - CONFIG['robot']['start_y_offset']
CONFIG['objects']['spawnable_colors'] = CONFIG['objects']['target_colors'] + ["gray"]
CONFIG['objects']['color_map'][CONFIG['objects']['other_bin_color_name']] = CONFIG['objects']['color_map'].get("other", (200, 200, 200))

def get_color_map():
    return CONFIG['objects']['color_map']

def get_visual_color(color_name):
    return get_color_map().get(color_name.lower(), CONFIG['objects']['color_map']['unknown'])

def generate_bin_definitions(config):
    target_colors = config['objects']['target_colors']
    other_bin_name = config['objects']['other_bin_color_name']
    bin_config = config['bins']
    game_area_width = config['display']['game_area_width']
    screen_height = config['display']['screen_height']
    color_map = get_color_map()

    bin_defs = []
    num_target_bins = len(target_colors)
    num_total_bins = num_target_bins + 1

    max_bins_per_row = bin_config['max_per_row']
    num_rows = math.ceil(num_total_bins / max_bins_per_row)

    bin_width = bin_config['width']
    bin_height = bin_config['height']

    effective_max_bins = min(num_total_bins, max_bins_per_row)
    total_bin_width = effective_max_bins * bin_width
    available_h_space = game_area_width - total_bin_width
    h_spacing = (available_h_space / (effective_max_bins + 1)) * bin_config['h_spacing_factor'] if effective_max_bins > 0 else 0
    h_spacing = max(10, h_spacing)

    v_spacing = bin_config['v_spacing']
    start_y = bin_config['top_margin']

    bin_id_counter = 0
    all_target_colors_set = set(target_colors)

    for r in range(num_rows):
        bins_in_this_row = max_bins_per_row if (r + 1) * max_bins_per_row <= num_total_bins else num_total_bins % max_bins_per_row
        if bins_in_this_row == 0 and num_total_bins > 0 : bins_in_this_row = max_bins_per_row

        current_row_width = bins_in_this_row * bin_width + max(0, bins_in_this_row - 1) * h_spacing
        current_start_x = (game_area_width - current_row_width) / 2

        for c in range(bins_in_this_row):
            bin_x = current_start_x + c * (bin_width + h_spacing)
            bin_y = start_y + r * (bin_height + v_spacing)

            if bin_id_counter < num_target_bins:
                target_color = target_colors[bin_id_counter]
                rule = lambda features, tc=target_color: features.get("color") == tc
                desc = f"Bin {bin_id_counter}: {target_color.capitalize()}"
                base_vis_color = get_visual_color(target_color)
                vis_color = tuple(min(255, x + bin_config['highlight_factor']) for x in base_vis_color)

            else:
                target_color = other_bin_name
                rule = lambda features, tcs=all_target_colors_set: features.get("color") not in tcs
                desc = f"Bin {bin_id_counter}: Other"
                vis_color = get_visual_color(other_bin_name)

            bin_defs.append({
                "id": bin_id_counter,
                "x": bin_x, "y": bin_y, "width": bin_width, "height": bin_height,
                "color": vis_color,
                "target_color": target_color,
                "rule_description": desc,
                "rule": rule,
                "center_x": bin_x + bin_width / 2,
                "center_y": bin_y + bin_height / 2,
            })
            bin_id_counter += 1

            if bin_id_counter >= num_total_bins: break
        if bin_id_counter >= num_total_bins: break

    return bin_defs

class ColorBinPredictor:
    def __init__(self, bin_definitions, config):
        self.bin_defs = {b['id']: b for b in bin_definitions}
        self.ai_config = config['ai']
        self.obj_config = config['objects']

        self.learning_rate = self.ai_config['learning_rate']
        self.decay_rate = self.ai_config['decay_rate']
        self.min_preference = self.ai_config['min_preference']
        self.default_preference = self.ai_config['default_preference']

        self.bin_preferences = {b_id: self.default_preference for b_id in self.bin_defs}
        self.other_bin_id = next((b['id'] for b in bin_definitions if b['target_color'] == self.obj_config['other_bin_color_name']), -1)
        self.target_colors_set = set(self.obj_config['target_colors'])

    def predict_bin(self, features):
        obj_color = features.get("color", "unknown")
        best_bin = self.other_bin_id
        max_score = -1

        for bin_id, bin_def in self.bin_defs.items():
            target_color = bin_def['target_color']
            preference = self.bin_preferences[bin_id]
            score = 0

            is_match = False
            if bin_id == self.other_bin_id:
                is_match = (obj_color not in self.target_colors_set)
            else:
                is_match = (obj_color == target_color)

            if is_match:
                score = preference

            if score > max_score:
                max_score = score
                best_bin = bin_id

        if best_bin == -1:
            print(f"Warning: No suitable bin found for object with color '{obj_color}'. Defaulting.")
            best_bin = self.other_bin_id if self.other_bin_id != -1 else 0

        return best_bin

    def update_preferences(self, was_correct, features, predicted_bin_id, correct_bin_id):
        update_factor = self.learning_rate
        obj_color = features.get("color", "unknown")

        def is_relevant_for_bin(bin_id, color):
            bin_def = self.bin_defs.get(bin_id)
            if not bin_def: return False
            if bin_id == self.other_bin_id:
                return color not in self.target_colors_set
            else:
                return color == bin_def['target_color']

        if not was_correct and predicted_bin_id is not None:
             if is_relevant_for_bin(predicted_bin_id, obj_color):
                 self.bin_preferences[predicted_bin_id] -= update_factor * self.bin_preferences[predicted_bin_id]

        if correct_bin_id is not None:
            if is_relevant_for_bin(correct_bin_id, obj_color):
                 reward_boost = 1.0 if not was_correct else 0.1
                 self.bin_preferences[correct_bin_id] += update_factor * reward_boost * (1.0 - self.bin_preferences[correct_bin_id])

        for bin_id in self.bin_preferences:
            self.bin_preferences[bin_id] *= (1.0 - self.decay_rate)
            self.bin_preferences[bin_id] = max(self.min_preference, min(1.0, self.bin_preferences[bin_id]))


class SimObject:
    def __init__(self, id, x, y, type, color="gray", weight=0.5, roughness=0.5, size=12):
        self.id = id
        self.x = x
        self.y = y
        self.type = type
        self.color_name = color.lower()
        self.weight = float(weight)
        self.roughness = float(roughness)
        self.size = int(size)
        self.picked_up = False
        self.visual_color = get_visual_color(self.color_name)
        self.predicted_bin_debug = None

    def get_features(self):
        return {
            "color": self.color_name,
            "weight": self.weight,
            "roughness": self.roughness,
            "size": self.size,
            "type": self.type
        }

    def draw(self, screen):
        if not self.picked_up:
            pygame.draw.circle(screen, self.visual_color, (int(self.x), int(self.y)), self.size)
            pygame.draw.circle(screen, (50, 50, 50), (int(self.x), int(self.y)), self.size, 1)

    def draw_held(self, screen, robot_x, robot_y, robot_angle):
         offset_dist = self.size + 10
         held_x = int(robot_x + offset_dist * math.cos(robot_angle))
         held_y = int(robot_y + offset_dist * math.sin(robot_angle))
         pygame.draw.circle(screen, self.visual_color, (held_x, held_y), self.size)
         pygame.draw.circle(screen, (50, 50, 50), (held_x, held_y), self.size, 1)


class Robot:
    def __init__(self, env, config):
        self.env = env
        self.config = config['robot']
        self.sim_config = config

        self.x = self.config['start_x']
        self.y = self.config['start_y']
        self.angle = -math.pi / 2
        self.held_object = None
        self.status = "Idle"
        self.speed = self.config['speed']
        self.pickup_range = self.config['pickup_range']
        self.drop_range = self.config['drop_range']
        self.visual_color = self.config['color']
        self.radius = self.config['radius']

        self.bin_predictor = ColorBinPredictor(env.bins, config)

        self.task_queue = []
        self.current_task = None
        self.target_x = self.x
        self.target_y = self.y
        self.last_dropped_object = None
        self.last_predicted_bin = None

        self.stats = {"correct_sorts": 0, "incorrect_sorts": 0, "total_sorted": 0, "steps_taken": 0, "tasks_failed": 0}

    def assign_sort_task(self, target_object):
        if isinstance(target_object, SimObject) and not target_object.picked_up:
            is_already_tasked = False
            if self.current_task and self.current_task[0] == "NAV_PICKUP" and self.current_task[1] == target_object:
                is_already_tasked = True
            if not is_already_tasked:
                is_already_tasked = any(t[0] == "NAV_PICKUP" and t[1] == target_object for t in self.task_queue)

            if not is_already_tasked:
                self.task_queue.append(("NAV_PICKUP", target_object))

    def update(self, dt):
        self.stats["steps_taken"] += 1

        if self.current_task is None and self.task_queue:
            self.current_task = self.task_queue.pop(0)
            action, target = self.current_task
            target_id_str = target if isinstance(target, int) else target.id

            if action == "NAV_PICKUP":
                self.status = f"Moving to pick {target_id_str}"
                self.target_x, self.target_y = target.x, target.y
            elif action == "NAV_DROP":
                 self.status = f"Moving to drop in bin {target_id_str}"
                 target_coords = self.env.get_bin_location(target)
                 if target_coords:
                     self.target_x, self.target_y = target_coords
                 else:
                     print(f"Error: Invalid bin index {target_id_str} for NAV_DROP.")
                     self.fail_task("Invalid Bin Index")
                     return

        if self.status.startswith("Moving"):
            self._move_towards_target(dt)

        self._check_task_completion()

    def _move_towards_target(self, dt):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.hypot(dx, dy)

        arrival_threshold = 1.0
        if distance < arrival_threshold:
             self.x = self.target_x
             self.y = self.target_y
        else:
            self.angle = math.atan2(dy, dx)
            move_dist = self.speed * dt
            if move_dist >= distance:
                self.x = self.target_x
                self.y = self.target_y
            else:
                self.x += (dx / distance) * move_dist
                self.y += (dy / distance) * move_dist

    def _check_task_completion(self):
        if self.current_task is None: return

        action, target = self.current_task
        distance_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)

        completion_threshold = 0
        if action == "NAV_PICKUP":
            completion_threshold = self.pickup_range / 2
        elif action == "NAV_DROP":
            completion_threshold = self.drop_range / 2

        if distance_to_target < completion_threshold:
            if action == "NAV_PICKUP":
                self.perform_pickup(target)
            elif action == "NAV_DROP":
                self.perform_drop(target)

    def perform_pickup(self, obj_to_pick):
        if self.held_object:
             print(f"Warning: Tried to pick up {obj_to_pick.id} but already holding {self.held_object.id}.")
             self.fail_task("Already Holding Object")
             return

        distance = math.hypot(self.x - obj_to_pick.x, self.y - obj_to_pick.y)
        if distance < self.pickup_range and not obj_to_pick.picked_up:
            self.held_object = obj_to_pick
            obj_to_pick.picked_up = True
            self.env.remove_object_from_active(obj_to_pick)
            self.status = f"Picked up {obj_to_pick.id}"

            features = self.held_object.get_features()
            predicted_bin_id = self.bin_predictor.predict_bin(features)
            self.last_predicted_bin = predicted_bin_id
            self.held_object.predicted_bin_debug = predicted_bin_id

            print(f"Robot Info: AI predicts Bin {predicted_bin_id} ({self.env.get_bin_target_color(predicted_bin_id)}) for {obj_to_pick.id} ({features['color']})")

            self.task_queue.insert(0, ("NAV_DROP", predicted_bin_id))
            self.current_task = None
        else:
            reason = "Out of Range" if distance >= self.pickup_range else "Object Already Picked Up?"
            print(f"Warning: Pickup failed for {obj_to_pick.id}. Dist: {distance:.1f}, Range: {self.pickup_range}. Reason: {reason}")
            self.fail_task("Pickup Range Failed")

    def perform_drop(self, predicted_bin_id):
         if not self.held_object:
              print("Error: Tried to drop but holding nothing.")
              self.fail_task("Nothing to Drop")
              return

         target_coords = self.env.get_bin_location(predicted_bin_id)
         if not target_coords:
             print(f"Error: Invalid target coordinates for bin {predicted_bin_id}.")
             self.fail_task(f"Invalid Bin {predicted_bin_id} for Drop")
             if self.held_object:
                  self.held_object.picked_up = False
                  self.held_object.x = self.x + random.uniform(-10, 10)
                  self.held_object.y = self.y + random.uniform(-10, 10)
                  self.env.add_object_to_active(self.held_object)
                  self.held_object = None
             return

         distance = math.hypot(self.x - target_coords[0], self.y - target_coords[1])
         if distance < self.drop_range:
             dropped_obj = self.held_object
             self.held_object = None
             dropped_obj.picked_up = False

             self.env.place_object_in_bin_area(dropped_obj, predicted_bin_id)
             self.last_dropped_object = dropped_obj
             self.status = f"Dropped {dropped_obj.id} in bin {predicted_bin_id}"

             self.env.check_object_placement(self.last_dropped_object, predicted_bin_id)
             self.current_task = None
         else:
              print(f"Warning: Drop failed for bin {predicted_bin_id}. Dist: {distance:.1f} > Range: {self.drop_range}")
              self.fail_task("Drop Range Failed")

    def receive_feedback(self, was_correct, features, correct_bin_id):
        predicted_bin_id = self.last_predicted_bin

        print(f"Robot Info: Feedback received - Correct: {was_correct}. Predicted: Bin {predicted_bin_id}, Correct: Bin {correct_bin_id}")

        self.bin_predictor.update_preferences(was_correct, features, predicted_bin_id, correct_bin_id)

        self.stats["total_sorted"] += 1
        if was_correct:
            self.stats["correct_sorts"] += 1
        else:
            self.stats["incorrect_sorts"] += 1

        self.last_dropped_object = None
        self.last_predicted_bin = None

    def fail_task(self, reason="Unknown"):
         print(f"Error: Task Failed - {reason}. Current task: {self.current_task}")
         self.stats["tasks_failed"] += 1

         if self.held_object:
             print(f" Dropping held object {self.held_object.id} due to failure.")
             self.held_object.picked_up = False
             drop_x = self.x + random.uniform(-self.radius * 1.5, self.radius * 1.5)
             drop_y = self.y + random.uniform(-self.radius * 1.5, self.radius * 1.5)
             game_w = self.sim_config['display']['game_area_width']
             game_h = self.sim_config['display']['screen_height']
             self.held_object.x = max(self.held_object.size, min(game_w - self.held_object.size, drop_x))
             self.held_object.y = max(self.held_object.size, min(game_h - self.held_object.size, drop_y))

             self.env.add_object_to_active(self.held_object)
             self.held_object = None

         self.current_task = None
         self.status = f"Failed: {reason}"

    def draw(self, screen):
        pygame.draw.circle(screen, self.visual_color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (30, 30, 30), (int(self.x), int(self.y)), self.radius, 2)
        end_x = self.x + self.radius * math.cos(self.angle)
        end_y = self.y + self.radius * math.sin(self.angle)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 3)

        if self.held_object:
             self.held_object.draw_held(screen, self.x, self.y, self.angle)


class Environment:
    def __init__(self, config):
        self.config = config
        self.width = config['display']['game_area_width']
        self.height = config['display']['screen_height']

        self.objects = []
        self.active_objects = []
        self.robot = None
        self.bins = generate_bin_definitions(config)
        self.bin_map = {b['id']: b for b in self.bins}
        self.objects_in_bins = {b["id"]: [] for b in self.bins}

        self.other_bin_id = next((b['id'] for b in self.bins if b['target_color'] == config['objects']['other_bin_color_name']), -1)
        self.target_colors_set = set(config['objects']['target_colors'])

    def add_robot(self, robot):
        self.robot = robot

    def add_object(self, obj):
        if isinstance(obj, SimObject):
             self.objects.append(obj)
             if not obj.picked_up:
                 self.active_objects.append(obj)

    def add_object_to_active(self, obj):
         if obj in self.objects and obj not in self.active_objects:
             obj.picked_up = False
             obj.x = max(obj.size, min(self.width - obj.size, obj.x))
             obj.y = max(obj.size, min(self.height - obj.size, obj.y))
             self.active_objects.append(obj)
             print(f"Debug: Object {obj.id} re-added to active objects.")

    def remove_object_from_active(self, obj):
        if obj in self.active_objects:
             self.active_objects.remove(obj)

    def place_object_in_bin_area(self, obj, bin_id):
        target_bin = self.bin_map.get(bin_id)
        if target_bin and isinstance(obj, SimObject):
            self.objects_in_bins[bin_id].append(obj)

            padding = obj.size + 2
            min_bin_x = target_bin['x'] + padding
            max_bin_x = target_bin['x'] + target_bin['width'] - padding
            min_bin_y = target_bin['y'] + padding
            max_bin_y = target_bin['y'] + target_bin['height'] - padding

            obj.x = random.uniform(min_bin_x, max_bin_x) if max_bin_x > min_bin_x else target_bin['center_x']
            obj.y = random.uniform(min_bin_y, max_bin_y) if max_bin_y > min_bin_y else target_bin['center_y']

            obj.picked_up = False

            print(f"Environment: Object {obj.id} ({obj.color_name}) placed in bin {bin_id}")
        else:
             print(f"Error: Could not place object {obj.id} in invalid bin id {bin_id}.")
             if self.robot:
                 obj.x = self.robot.x + random.uniform(-5, 5)
                 obj.y = self.robot.y + random.uniform(-5, 5)
             else:
                 obj.x = self.width / 2
                 obj.y = self.height / 2
             self.add_object_to_active(obj)

    def get_bin_location(self, bin_id):
        target_bin = self.bin_map.get(bin_id)
        return (target_bin['center_x'], target_bin['center_y']) if target_bin else None

    def get_bin_target_color(self, bin_id):
        target_bin = self.bin_map.get(bin_id)
        return target_bin['target_color'] if target_bin else "Unknown"

    def get_correct_bin(self, obj_features):
        obj_color = obj_features.get("color")
        correct_bin_id = -1

        for b in self.bins:
            if b['id'] != self.other_bin_id and b['rule'](obj_features):
                 correct_bin_id = b['id']
                 break

        if correct_bin_id == -1 and self.other_bin_id != -1:
            other_bin = self.bin_map[self.other_bin_id]
            if other_bin['rule'](obj_features):
                correct_bin_id = self.other_bin_id

        if correct_bin_id == -1:
            print(f"Warning: No matching bin rule found for object features {obj_features}. Defaulting.")
            correct_bin_id = self.other_bin_id if self.other_bin_id != -1 else 0

        return correct_bin_id

    def check_object_placement(self, dropped_object, predicted_bin_id):
        if dropped_object is None:
            print("Error: check_object_placement called with None object.")
            return

        features = dropped_object.get_features()
        correct_bin_id = self.get_correct_bin(features)
        was_correct = (predicted_bin_id == correct_bin_id)

        pred_bin_color = self.get_bin_target_color(predicted_bin_id)
        corr_bin_color = self.get_bin_target_color(correct_bin_id)
        result_str = 'Correct' if was_correct else 'Incorrect'
        print(f"Environment Check: Obj {dropped_object.id} ({features['color']}) "
              f"placed in Bin {predicted_bin_id} ({pred_bin_color}). "
              f"Correct: Bin {correct_bin_id} ({corr_bin_color}). Result: {result_str}")

        if self.robot:
             self.robot.receive_feedback(was_correct, features, correct_bin_id)
        else:
             print("Error: Robot reference missing in environment for feedback.")

    def update(self, dt):
        pass

    def draw(self, screen):
        bin_font = pygame.font.SysFont(None, 18)

        for b in self.bins:
            pygame.draw.rect(screen, b['color'], (b['x'], b['y'], b['width'], b['height']))
            pygame.draw.rect(screen, (50, 50, 50), (b['x'], b['y'], b['width'], b['height']), 2)
            id_text = bin_font.render(f"Bin {b['id']} ({b['target_color'].capitalize()})", True, CONFIG['display']['text_color'])
            screen.blit(id_text, (b['x'] + 5, b['y'] + 5))

        for obj in self.active_objects:
            obj.draw(screen)

        for bin_id, object_list in self.objects_in_bins.items():
            for obj in object_list:
                 obj.draw(screen)


class Simulation:
    def __init__(self, config):
        pygame.init()
        pygame.font.init()

        self.config = config
        self.display_config = config['display']
        self.sim_config = config['simulation']

        self.screen_width = self.display_config['screen_width']
        self.screen_height = self.display_config['screen_height']
        self.game_area_width = self.display_config['game_area_width']
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Dynamic Sorting Simulation")
        self.clock = pygame.time.Clock()

        try:
            self.main_font = pygame.font.SysFont('Arial', 26)
            self.info_font = pygame.font.SysFont('Arial', 20)
            self.small_font = pygame.font.SysFont('Arial', 16)
        except Exception as e:
            print(f"Warning: SysFont 'Arial' failed ({e}), using default font.")
            self.main_font = pygame.font.Font(None, 28)
            self.info_font = pygame.font.Font(None, 22)
            self.small_font = pygame.font.Font(None, 18)

        self.paused = False
        self.running = True
        self.environment = None
        self.robot = None
        self.setup_simulation()

    def setup_simulation(self):
        print("Setting up new simulation environment...")
        self.environment = Environment(self.config)
        self.robot = Robot(self.environment, self.config)
        self.environment.add_robot(self.robot)

        self.spawn_objects(self.sim_config['num_objects_to_spawn'])
        self.assign_tasks_for_active_objects()
        print(f"Setup complete. {len(self.robot.task_queue)} initial tasks assigned.")
        self.paused = False

    def assign_tasks_for_active_objects(self):
        assigned_count = 0
        targeted_object_ids = set()
        if self.robot.current_task and self.robot.current_task[0] == "NAV_PICKUP":
             targeted_object_ids.add(self.robot.current_task[1].id)
        for task_type, target in self.robot.task_queue:
             if task_type == "NAV_PICKUP":
                 targeted_object_ids.add(target.id)

        for obj in list(self.environment.active_objects):
            if not obj.picked_up and obj.id not in targeted_object_ids:
                self.robot.assign_sort_task(obj)
                targeted_object_ids.add(obj.id)
                assigned_count += 1

        if assigned_count > 0:
            print(f"Assigned {assigned_count} new sorting tasks.")

    def spawn_objects(self, num_objects):
        spawn_count = 0
        obj_config = self.config['objects']
        robot_config = self.config['robot']
        display_config = self.config['display']
        sim_conf = self.config['simulation']

        bin_bottom_y = 0
        if self.environment.bins:
             bin_bottom_y = max(b['y'] + b['height'] for b in self.environment.bins)

        buffer = sim_conf['initial_spawn_buffer']
        min_y = bin_bottom_y + buffer
        max_y = robot_config['start_y'] - self.robot.radius - buffer
        min_x = buffer
        max_x = display_config['game_area_width'] - buffer

        if min_y >= max_y or min_x >= max_x:
             print(f"Error: Calculated spawn area is invalid (min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}). "
                   "Check configuration for bin layout, robot start, and buffer.")
             min_y = display_config['screen_height'] * 0.2
             max_y = display_config['screen_height'] * 0.6
             min_x = display_config['game_area_width'] * 0.1
             max_x = display_config['game_area_width'] * 0.9
             print(f"Using fallback spawn area: x=[{min_x:.0f}-{max_x:.0f}], y=[{min_y:.0f}-{max_y:.0f}]")
             if min_y >= max_y or min_x >= max_x:
                 print("Error: Fallback spawn area is also invalid. Cannot spawn objects.")
                 return

        print(f"Spawning {num_objects} objects in area: x=[{min_x:.0f}-{max_x:.0f}], y=[{min_y:.0f}-{max_y:.0f}]")

        for i in range(num_objects):
            obj_color = random.choice(obj_config['spawnable_colors'])
            obj_weight = round(random.uniform(obj_config['min_weight'], obj_config['max_weight']), 2)
            obj_roughness = round(random.uniform(obj_config['min_roughness'], obj_config['max_roughness']), 2)
            obj_size = random.uniform(obj_config['min_size'], obj_config['max_size'])
            obj_x = random.uniform(min_x + obj_size, max_x - obj_size)
            obj_y = random.uniform(min_y + obj_size, max_y - obj_size)
            obj_id = f"obj_{time.time():.4f}_{spawn_count}"

            obj = SimObject(id=obj_id, x=obj_x, y=obj_y,
                            type=random.choice(obj_config['types']),
                            color=obj_color, weight=obj_weight,
                            roughness=obj_roughness, size=obj_size)
            self.environment.add_object(obj)
            spawn_count += 1

        print(f"Spawned {spawn_count} objects.")

    def run(self):
        while self.running:
            dt = min(self.clock.tick(self.display_config['fps']) / 1000.0, 0.05)

            self.handle_events()

            if not self.paused:
                self.update(dt)

            self.draw()

        pygame.quit()
        print("Simulation exited.")
        self.print_final_stats()

    def print_final_stats(self):
         print("\n--- Final Simulation Stats ---")
         if hasattr(self, 'robot') and self.robot:
              stats = self.robot.stats
              for key, value in stats.items():
                  key_title = key.replace('_', ' ').title()
                  print(f"{key_title}: {value}")

              total = stats['total_sorted']
              correct = stats['correct_sorts']
              accuracy = (correct / total * 100) if total > 0 else 0
              print(f"Final Accuracy: {accuracy:.2f}%")

              print("\nFinal AI Bin Preferences:")
              if hasattr(self.robot, 'bin_predictor') and self.robot.bin_predictor:
                  prefs = self.robot.bin_predictor.bin_preferences
                  bin_defs_map = self.robot.bin_predictor.bin_defs
                  for bin_id in sorted(prefs.keys()):
                       bin_label = bin_defs_map.get(bin_id, {}).get('target_color', 'Unknown').capitalize()
                       print(f" Bin {bin_id} ({bin_label}): {prefs[bin_id]:.4f}")
              else:
                  print(" (Bin predictor data not available)")
         else:
             print("(Robot instance not found or not initialized)")
         print("----------------------------")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Simulation {'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    print("Resetting simulation...")
                    self.setup_simulation()
                elif event.key == pygame.K_s:
                     if not self.paused:
                         print(f"Spawning {self.sim_config['respawn_count']} additional objects...")
                         self.spawn_objects(self.sim_config['respawn_count'])
                         self.assign_tasks_for_active_objects()
                     else:
                         print("Cannot spawn objects while paused.")

    def update(self, dt):
        self.robot.update(dt)
        self.environment.update(dt)

        if self.robot.status == "Idle" and not self.robot.task_queue and self.environment.active_objects:
             self.assign_tasks_for_active_objects()

    def draw(self):
        self.screen.fill(self.display_config['background_color'])
        game_area_rect = pygame.Rect(0, 0, self.game_area_width, self.screen_height)
        self.environment.draw(self.screen)
        self.robot.draw(self.screen)

        line_color = (100, 100, 100)
        pygame.draw.line(self.screen, line_color,
                         (self.game_area_width, 0),
                         (self.game_area_width, self.screen_height), 2)

        self.draw_info_panel()

        if self.paused:
            self.draw_pause_overlay()

        pygame.display.flip()

    def draw_pause_overlay(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((200, 200, 200, 170))

        pause_text = self.main_font.render("PAUSED", True, (50, 50, 50))
        controls_text = self.info_font.render("(P: Resume, R: Reset, S: Spawn, Esc: Quit)", True, (50, 50, 50))

        pause_rect = pause_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 20))
        controls_rect = controls_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 20))

        overlay.blit(pause_text, pause_rect)
        overlay.blit(controls_text, controls_rect)
        self.screen.blit(overlay, (0, 0))

    def draw_info_panel(self):
        panel_x_start = self.game_area_width + 10
        current_y = 20
        line_spacing = 6
        indent = 15
        text_color = self.display_config['text_color']
        error_color = self.display_config['error_color']

        def draw_text_line(text, font, y, color=text_color, bold=False, indent_level=0):
            original_bold = font.get_bold()
            font.set_bold(bold)
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect(topleft=(panel_x_start + indent * indent_level, y))
            self.screen.blit(text_surface, text_rect)
            font.set_bold(original_bold)
            return y + font.get_height() + line_spacing

        current_y = draw_text_line("Simulation Status", self.main_font, current_y, bold=True)
        status_color = error_color if "Failed" in self.robot.status else text_color
        current_y = draw_text_line(f"Robot: {self.robot.status}", self.info_font, current_y, color=status_color, indent_level=1)

        tasks_left = len(self.robot.task_queue)
        current_task_str = "None"
        if self.robot.current_task:
             action, target = self.robot.current_task
             target_id_str = target.id if isinstance(target, SimObject) else f"Bin {target}"
             current_task_str = f"{action} -> {target_id_str}"
        current_y = draw_text_line(f"Tasks in Queue: {tasks_left}", self.info_font, current_y, indent_level=1)
        current_y = draw_text_line(f"Current Task: {current_task_str}", self.small_font, current_y, indent_level=2)
        current_y += line_spacing

        current_y = draw_text_line("Performance", self.main_font, current_y, bold=True)
        stats = self.robot.stats
        total = stats['total_sorted']
        correct = stats['correct_sorts']
        incorrect = stats['incorrect_sorts']
        failed = stats['tasks_failed']
        accuracy = (correct / total * 100) if total > 0 else 0
        current_y = draw_text_line(f"Total Sorted: {total}", self.info_font, current_y, indent_level=1)
        current_y = draw_text_line(f"Correct: {correct}", self.info_font, current_y, indent_level=2, color=(0, 150, 0) if correct > 0 else text_color)
        current_y = draw_text_line(f"Incorrect: {incorrect}", self.info_font, current_y, indent_level=2, color=error_color if incorrect > 0 else text_color)
        current_y = draw_text_line(f"Accuracy: {accuracy:.1f}%", self.info_font, current_y, indent_level=1, bold=True)
        fail_color = error_color if failed > 0 else text_color
        current_y = draw_text_line(f"Tasks Failed: {failed}", self.info_font, current_y, color=fail_color, indent_level=1)
        current_y += line_spacing

        current_y = draw_text_line("AI Bin Preferences", self.main_font, current_y, bold=True)
        if hasattr(self.robot, 'bin_predictor') and self.robot.bin_predictor:
            prefs = self.robot.bin_predictor.bin_preferences
            bin_defs_map = self.robot.bin_predictor.bin_defs
            for bin_id in sorted(prefs.keys()):
                 bin_label = bin_defs_map.get(bin_id, {}).get('target_color', '?').capitalize()
                 pref_value = prefs[bin_id]
                 pref_color_g = int(50 + 150 * pref_value)
                 pref_color_r = int(50 + 100 * (1-pref_value))
                 pref_color = (pref_color_r, pref_color_g, 50)

                 current_y = draw_text_line(f"Bin {bin_id} ({bin_label}): {pref_value:.3f}",
                                            self.small_font, current_y, indent_level=1, color=pref_color)
        else:
            current_y = draw_text_line("N/A", self.small_font, current_y, indent_level=1)
        current_y += line_spacing

        controls_y_start = self.screen_height - 100
        controls_y_start = draw_text_line("Controls", self.main_font, controls_y_start, bold=True)
        controls_y_start = draw_text_line("P : Pause/Resume", self.small_font, controls_y_start)
        controls_y_start = draw_text_line("R : Reset Simulation", self.small_font, controls_y_start)
        controls_y_start = draw_text_line("S : Spawn Objects", self.small_font, controls_y_start)
        controls_y_start = draw_text_line("Esc : Quit", self.small_font, controls_y_start)

if __name__ == '__main__':
    try:
        sim = Simulation(CONFIG)
        sim.run()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---", file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        import traceback
        print("\n--- Traceback ---", file=sys.stderr)
        traceback.print_exc()
        print("-----------------", file=sys.stderr)
        if pygame.get_init():
            pygame.quit()
        sys.exit(1)