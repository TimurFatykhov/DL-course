import numpy as np
import time

def __getkey__(seed):
    np.random.seed(int(seed))
    key = np.random.randint(1001, 10000)
    return key

def test1(func):
    "функция суммы двух чисел"
    inp = (5, 10)
    ret = func(*inp)
    if not ret == 15:
        print(f"Ошибка! Входные параметры: {inp}. Результат: {ret}. Ожидалось: 15")
        return
    print(f'Тесты пройдены! Ключ: {__getkey__(ret)}')
    
    
def test2(func):
    "проверка на четность"
    inp = 6
    ret = func(inp)
    if not ret == "четное":
        print(f"Ошибка! Входные параметры: {inp}. Результат: {ret}. Ожидалось: четное")
        return
    inp = 5
    ret = func(inp)
    if not ret == "нечетное":
        print(f"Ошибка! Входные параметры: {inp}. Результат: {ret}. Ожидалось: нечетное")
        return
    print(f'Тесты пройдены! Ключ: {__getkey__(len(ret))}')
    
    
def test3(func):
    "сумма элементов списка"
    inp = [-1, 2, 10, -5, 8, 11]
    ret = func(inp)
    if not ret == 25:
        print(f"Ошибка! Входные параметры: {inp}. Результат: {ret}. Ожидалось: 25")
        return
    print(f'Тесты пройдены! Ключ: {__getkey__(ret)}')
    
    
def test4(func):
    "все отрицательные элементы списка заменяются на нули"
    np.random.seed(int(time.time()))
    inp = list(np.random.randint(-10, 10, 12))
    ret = func(inp)
    res = list(np.clip(inp, 0, 10))
    if not ret == res:
        print(f"Ошибка! Входные параметры: {inp}. Результат: {ret}. Ожидалось: {res}")
        return
    print(f'Тесты пройдены! Ключ: {__getkey__(np.median(func(list(np.arange(-10, 15)))))}')
    
    
def test5(func):
    "вычисляет скалярное произведение двух списков"
    np.random.seed(int(time.time()))
    for _ in range(500):
        inp1 = list(np.random.randint(-3, 10, 4))
        inp2 = list(np.random.randint(-3, 10, 4))
        lim = np.random.randint(-10, 100)
        ret = None
        if np.random.rand() < 0.5:
            ret = func(inp1, inp2)
            lim = 100
        else:
            ret = func(inp1, inp2, lim)
        res = np.dot(inp1, inp2) < lim
        if not ret == res:
            print(f"Вычислительная ошибка! Входные параметры: {inp1, inp2, lim}. Результат: {ret}. Ожидалось: {res}")
            return
    try:
        ret = func([1,1], [1,1,1])
    except:
        print(f"Ошибка исполнения! Входные параметры: {[1,1], [1,1,1]}. Функция остановила свою работу из-за ошибки.")
        return
    
    if not ret == 'некорректный ввод':
        print(f"Ошибка! Входные параметры: {[1,1], [1,1,1]}. Результат: {ret}. Ожидалось: некорректный ввод")
        return
    print(f'Тесты пройдены! Ключ: {__getkey__(100 + func([1,1], [100, 100]))}')
    
    
    