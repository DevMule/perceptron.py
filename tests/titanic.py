import csv
import numpy as np
from perceptron import Rumelhart

appeal_weights = {  # I dunno what i wanted it to look like #1
    " Lady.": [10, 30],
    " the Countess.": [10, 30],
    " Jonkheer.": [10, 30],
    " Miss.": [10, 30],
    " Ms.": [10, 30],
    " Mme.": [10, 30],
    " Major.": [9, 35],
    " Mrs.": [9, 30],
    " Rev.": [9, 30],
    " Sir.": [8, 30],
    " Master.": [8, 30],
    " Don.": [7, 30],
    " Dr.": [5, 30],
    " Mr.": [5, 30],
    " Mlle.": [6, 30],
    " Capt.": [6, 30],
    " Col.": [5, 30],
}


def maturity(passenger):  # I dunno what i wanted it to look like #2
    name = passenger['Name']
    for key in appeal_weights:
        if key in name:
            appeal_w = appeal_weights[key][0]
            age_w = passenger['Age'] or appeal_weights[key][1]
            return [appeal_w, age_w]


def embarked(k):
    if k == 'S':
        return -1
    elif k == 'C':
        return 0
    else:
        return 1


inp_data = []
out_data = []
# максимальные и минимальные значения для перевода данных в вид [0; 1]
maxs = [-999999, -999999, -999999, -999999, -999999, -999999, -999999, -999999]
mins = [999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999]
# парсим данные
with open('titanic.csv') as f:
    passengers = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
for passenger in passengers:
    m = maturity(passenger)
    new_passenger = [float(passenger['Sex'] == 'male'),
                     float(passenger['Pclass']),
                     float(m[0]),
                     float(m[1]),
                     float(passenger['SibSp']),
                     float(passenger['Parch']),
                     float(passenger['Fare']),
                     float(embarked(passenger['Embarked']))]
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
network = Rumelhart(8, 16, 1)
# обучаем на всю выборку, кроме последних 100
network.learn(
    inp_data[0:total - test_count],
    out_data[0:total - test_count],
    epochs=100000,
    err_print_frequency=5000
)
# проверяем обучение на последних 100 людях
summary_true = 0
for i in range(total - test_count, total):
    predicted = round(network.feedforward(inp_data[i])[0])
    real = out_data[i][0]
    print('got:', predicted, ' real:', real)
    summary_true += int(predicted == real)
print("successfully predicted: ", summary_true, "/", test_count)  # successfully predicted:  82 / 100
