import tensorflow as tf
from genes import GeneModel, load_trace
import numpy as np
import pandas as pd

sess = tf.Session()
model = GeneModel(model_type="ltc", model_size=64)

model.restore()

# test_data = [0.          1.          0.          0.          0.          4.73312807
#   20.         24.          4.73312807  1.          0.          3.
#    0.          0.          0.          1.          0.          1.
#    3.          2.          2.          2.          0.          0.
#    3.          4.          0.          2.          2.          0.
#    5.9727478   2.          1.          0.          0.          0.
#    0.]

df = pd.read_csv("data/genes/train.csv")
x, y = load_trace(df)

correct = 0
for i in range(100):
    index = np.random.randint(0, 1500)
    test_data = x[index]

    test_data_arr = np.array(test_data, dtype=np.float32)
    test_data_arr = test_data_arr.reshape(1, -1, len(test_data))
    if(int(model.run(test_data_arr)) == y[index]):
        correct+=1
        
print(correct)


# test_data = [0, 1, 0, 0, 0, 4.73312807, 20, 24, 4.73312807, 1, 0, 3, 0, 0, 0, 1, 0, 1, 3, 2, 2, 2, 0, 0, 3, 4, 0, 2, 2, 0, 5.9727478, 2, 1, 0, 0, 0, 0,]

test_data = x[22]

test_data_arr = np.array(test_data, dtype=np.float32)
test_data_arr = test_data_arr.reshape(1, -1, len(test_data))


print(model.run(test_data_arr))
