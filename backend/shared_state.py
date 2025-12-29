# Simple in-memory state store
# Maps alert_id -> status ('active', 'acknowledged', 'dismissed')
ALERT_STATE = {}

def get_alert_status(alert_id):
    return ALERT_STATE.get(alert_id, 'active')

def set_alert_status(alert_id, status):
    ALERT_STATE[alert_id] = status

def reset_state():
    global ALERT_STATE
    ALERT_STATE = {}

# Simulation State
# Boolean: Is 'Optimized Mode' active?
SIMULATION_ACTIVE = False

def set_simulation_active(active: bool):
    global SIMULATION_ACTIVE
    SIMULATION_ACTIVE = active

def get_simulation_active():
    return SIMULATION_ACTIVE
