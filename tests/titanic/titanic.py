import csv

from perceptron import Perceptron

inp_data = []
out_data = []
# парсим данные
with open('data.csv') as f:
    passengers = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
for passenger in passengers:
    inp_data.append(
        [float(passenger['Sex'] == 'male'),
         float(passenger['Pclass']),
         float(passenger['Age'] or 0),
         float(passenger['SibSp']),
         float(passenger['Parch']),
         float(passenger['Fare'])]
    )
    out_data.append([float(passenger['Survived'])])
total = len(passengers)

network = Perceptron(6, 12, 10, 8, 1)
# обучаем на всю выборку, кроме последних 10
network.learn(
    inp=inp_data[0:total - 10],
    out=out_data[0:total - 10],
    epochs=100000,
    learn_rate=10,
    err_print_frequency=1000
)
# проверяем обучение на последних 10 людях
for i in range(total - 10, total):
    print('got:', round(network.feedforward(inp_data[i])[0], 5), ' real:', out_data[i][0])
