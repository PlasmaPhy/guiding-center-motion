import numpy as np
import gcmotion as gcm

print("NumPy:")
print(f"Name = {np.__name__}")
print(f"Path = {np.__path__}")

print()

print("GCMotion:")
print(f"Name = {gcm.__name__}")
print(f"Path = {gcm.__path__}")  # Installed locally so its editable
