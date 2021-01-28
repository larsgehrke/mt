import train
import test

mode = 1  # 1 -> "train" or 2 -> "test"

mode_str = "train" if mode == 1 else "test"

print("Running sript:", mode_str + ".py")

if mode_str == "train":
    train.run_training()
else:
    test.run_testing()
