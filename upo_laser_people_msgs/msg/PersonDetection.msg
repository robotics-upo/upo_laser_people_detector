# Detection class
uint32 type
uint32 TYPE_UNSPECIFIED=0
# Other types reserved for future use

# Detection score [0, 1]
float32 score

# Only X and Y components used; Z is ignored
geometry_msgs/Point position
