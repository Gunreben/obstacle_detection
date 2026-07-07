# Axis-aligned safety zone rectangle in the fusion target frame.
# NOTE: current deployment convention is x = lateral, y = forward
# (matches the mesh-derived TF tree; see package README).
string name        # "red" or "yellow"
float64 x_min
float64 x_max
float64 y_min
float64 y_max
