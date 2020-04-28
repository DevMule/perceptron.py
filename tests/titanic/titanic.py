import csv
import numpy as np
from perceptron import Perceptron

inp_data = []
out_data = []
# максимальные и минимальные значения для перевода данных в вид [0; 1]
maxs = [-999999, -999999, -999999, -999999, -999999, -999999]
mins = [999999, 999999, 999999, 999999, 999999, 999999]
# парсим данные
with open('data.csv') as f:
    passengers = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
for passenger in passengers:
    if not (passenger['Sex'] and passenger['Pclass'] and passenger['Age'] and passenger['SibSp'] and passenger[
        'Parch']) or not passenger['Fare'] or not passenger['Survived']: continue
    new_passenger = [float(passenger['Sex'] == 'male'),
                     float(passenger['Pclass']),
                     float(passenger['Age']),
                     float(passenger['SibSp']),
                     float(passenger['Parch']),
                     float(passenger['Fare'])]
    inp_data.append(new_passenger)
    out_data.append([float(passenger['Survived'])])
    for i in range(len(new_passenger)):
        maxs[i] = max(maxs[i], new_passenger[i])
        mins[i] = min(mins[i], new_passenger[i])

for passenger in inp_data:
    for i in range(len(passenger)):
        passenger[i] = (passenger[i] - mins[i]) / (maxs[i] - mins[i])

# ============================== обучение ==============================
total = len(inp_data)
test_count = 100  # count of data pairs which is not will be learned, but which will be tested
np.random.seed(2)
network = Perceptron(6, 4, 1)
# обучаем на всю выборку, кроме последних 100
network.learn(
    inp=inp_data[0:total - test_count],
    out=out_data[0:total - test_count],
    epochs=100000,
    learning_rate=.1,
    err_print_frequency=1000
)
# проверяем обучение на последних 100 людях
summary_true = 0
for i in range(total - test_count, total):
    predicted = round(network.feedforward(inp_data[i])[0])
    real = out_data[i][0]
    print('got:', predicted, ' real:', real)
    summary_true += int(predicted == real)
print("successfully predicted: ", summary_true, "/", test_count)  # successfully predicted:  78 / 100
