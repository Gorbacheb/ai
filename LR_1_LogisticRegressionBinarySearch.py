import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def read_data() -> tuple[int, int]:
    n_classes, n_elements = np.loadtxt('input.txt', delimiter=',')
    return int(n_classes), int(n_elements)


def write_data(data):
    with open('output.txt', 'w') as f:
        f.write(str(data))


def create_random_cloud(n_elements: int):
    n_elements = int(n_elements)
    cloud0 = np.random.randn(n_elements, 2)
    return cloud0


def create_dataset(n_classes: int, n_elements: int, cloud, distance: float) -> tuple[np.ndarray, list[int]]:
    n_elements = int(n_elements)
    cloud0 = cloud
    cloud1 = cloud + np.array([distance, distance])
    cloud2 = cloud + np.array([distance * 2, distance * 2])
    annotation1 = [0] * n_elements
    annotation2 = [1] * n_elements
    annotation3 = [2] * n_elements
    annotations = annotation1 + annotation2 + annotation3
    XY = np.vstack([cloud0, cloud1, cloud2])
    return XY, annotations


def split_dataset(X: np.ndarray, Y: list[int]) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
    data, labels = shuffle(X, Y)
    test_size = int(len(data) * 0.5)
    Xtrain = data[:test_size]
    Ytrain = labels[:test_size]
    Xtest = data[test_size:]
    Ytest = labels[test_size:]
    return Xtrain, Ytrain, Xtest, Ytest


def create_classifier():
    return LogisticRegression()


def train_clf(clf, x_train: np.ndarray, y_train: list[int]):
    clf.fit(x_train, y_train)


def test_clf(clf, x_test: np.ndarray, y_test: list[int]) -> float:
    predictions = clf.predict(x_test)
    return accuracy_score(y_test, predictions)


def run_pipeline(n_classes: int, n_elements: int, cloud, distance) -> float:
    X, Y = create_dataset(n_classes, n_elements, cloud, distance)
    x_train, y_train, x_test, y_test = split_dataset(X, Y)

    clf = create_classifier()
    train_clf(clf, x_train, y_train)
    accuracy = test_clf(clf, x_test, y_test)
    return accuracy


n_classes, n_elements = 3, 220#read_data()
cloud = create_random_cloud(n_elements)

left = 0.832
right = 0.834
l = 0.0
r = 10.0 #макс размер облака
distance = r
result = -1
# бинпоиск по дистанции между облаками, функция погрешности монотонна (нестрого) тк, при сближении облаков результат точно
# не лучше -> бинпоиск работает
bad_attempts = 0
while not(left <= result <= right) and bad_attempts < 100:
    if (r - l < 0.001): #если все плохо, то пытаемся с новым облаком
        cloud = create_random_cloud(n_elements)
        l = 0
        r = 10.0
        bad_attempts += 1
    m = (l + r) / 2
    result = run_pipeline(n_classes, n_elements, cloud, m)
    if result >= right:
        r = m
    if result <= left:
        l = m
if left <= result <= right:
    print(result)
else:
    print("Найти такое расстояние для заданного промежутка не получается")
#write_data(result)
