import time

class ViolationTracker:
    def __init__(self, tolerance_seconds: float, cooldown_seconds: float = 30.0):
        self.tolerance_seconds = tolerance_seconds
        self.cooldown_seconds = cooldown_seconds
        self.start_time = None
        self.last_alert_time = None
        self.is_tracking = False

    def update(self, is_bad: bool) -> str:
        """
        Updates the tracker with the current frame's status.
        Returns: 'NORMAL', 'WARNING', 'VIOLATION', or 'SUPPRESSED'
        """
        if not is_bad:
            self.reset()
            return "NORMAL"

        if not self.is_tracking:
            self.is_tracking = True
            self.start_time = time.time()
            return "WARNING" # Initial detection is a warning/start

        current_duration = time.time() - self.start_time
        
        if current_duration >= self.tolerance_seconds:
             now = time.time()
             # Check cooldown
             if self.last_alert_time is None or (now - self.last_alert_time > self.cooldown_seconds):
                 self.last_alert_time = now
                 return "VIOLATION"
             else:
                 return "SUPPRESSED"
        else:
            return "WARNING"

    def reset(self):
        self.is_tracking = False
        self.start_time = None
        # Do not reset last_alert_time, cooldown persists across brief normal states? 
        # Actually user said "keeps bombarding".
        # If I reset last_alert_time here, then looking away -> VIOLATION -> look back (reset) -> look away -> VIOLATION.
        # This allows "spamming" by head bobbing.
        # But "cool off period of 30s for each different type".
        # If I look away (Alert), then look back, then look away immediately. Should I alert again?
        # Usually yes, that's a new violation instance. A "cooldown" usually applies to *continuous* or *frequent* triggering.
        # If I look back, I am "good".
        # Let's keep last_alert_time persistent? 
        # If I keep it persistent, then: Alert -> Normal -> Alert (within 30s) -> Suppressed.
        # This prevents spamming via toggling. This is safer for "bombarding".
        # I will keep `last_alert_time` persistent.

