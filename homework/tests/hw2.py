import numpy as np

def __getkey__(seed):
    np.random.seed(int(seed))
    key = np.random.randint(1001, 10000)
    return key
    

def test1(func):
    "скалярное произведение векторов"
    for _ in range(10):
        inp1, inp2 = np.random.randint(-20, 20, (2, 7))
        try:
            ret = func(list(inp1), list(inp2))
        except:
            print(f"Ошибка исполнения! Входные параметры: {list(inp1), list(inp2)}. Функция остановила свою работу из-за ошибки.")
            return
            
        true = inp1.dot(inp2)
        if not ret == true:
            print(f"Ошибка вычислений! Входные параметры: {list(inp1), list(inp2)}. Результат: {ret}. Ожидалось: {true}")
            return
    np.random.seed(17)
    inp1 = list(np.random.randint(-5, 5, 3000))
    inp2 = list(np.random.randint(-5, 5, 3000))
    print(f'Тесты пройдены! Ключ: {__getkey__(func(inp1, inp2))}')
    

def test2(func):
    "извлечение столбца"
    for _ in range(10):
        shape = np.random.randint(2,5,2)
        inp1 = np.random.randint(-20, 20, shape)
        inp2 = np.random.randint(shape[1])
        try:
            ret = func(list(inp1), inp2)
        except:
            print(f"Ошибка исполнения! Входные параметры: {list(inp1), inp2}. Функция остановила свою работу из-за ошибки.")
            return
            
        true = list(inp1[:,inp2])
        if not np.array_equal(ret, true):
            print(f"Ошибка вычислений! Входные параметры: {list(inp1), inp2}. Результат: {ret}. Ожидалось: {true}")
            return
    np.random.seed(17)
    inp = list(np.random.randint(-10, 10, (50, 50)))
    col = np.random.randint(50)
    print(f'Тесты пройдены! Ключ: {__getkey__(np.max(func(inp, col)))}')
    

def test3(func):
    "транспонирование матрицы"
    for _ in range(10):
        shape = np.random.randint(2,5,2)
        inp1 = np.random.randint(-20, 20, shape)
        try:
            ret = func(list(inp1))
        except:
            print(f"Ошибка исполнения! Входные параметры: {inp1}. Функция остановила свою работу из-за ошибки.")
            return
            
        true = inp1.T
        if not np.array_equal(ret, true):
            print(f"Ошибка вычислений! Входные параметры: {list(inp1)}. Результат: {ret}. Ожидалось: {true}")
            return
    np.random.seed(17)
    inp = np.random.randint(2, 10, (50, 40))
    print(f'Тесты пройдены! Ключ: {__getkey__(func(list(inp))[10][49])}')
    

def test4(func):
    "перемножение матриц"
    for _ in range(10):
        shape = np.random.randint(2,5,3)
        inp1 = np.random.randint(-7, 7, shape[:-1])
        inp2 = np.random.randint(-7, 7, shape[1:])
        try:
            ret = func(list(inp1), list(inp2))
        except:
            print(f"Ошибка исполнения! Входные параметры: {inp1, inp2}. Функция остановила свою работу из-за ошибки.")
            return
            
        true = list(inp1.dot(inp2))
        if not np.array_equal(ret, true):
            print(f"Ошибка вычислений! Входные параметры: {list(inp1), list(inp2)}. Результат: {ret}. Ожидалось: {true}")
            return
    np.random.seed(17)
    inp1 = list(np.random.randint(-10, 10, (5, 4)))
    inp2 = list(np.random.randint(-10, 10, (4, 5)))
    print(f'Тесты пройдены! Ключ: {__getkey__(np.max(func(inp1, inp2)))}')
    