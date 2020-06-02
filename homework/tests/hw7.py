import numpy as np

x = np.array([[ 8,  5, -5, -1, -1,  7, -7,  0,  8,  3],
              [-2,  9,  8, -7,  0,  7, -8, -1, -2,  6],
              [-3,  2,  3,  1, -8,  6,  3,  2,  2, -2]])
w = np.array([[ -3,  -3,   9,   3,   5],
               [  9,   4,  -1,   2,   0],
               [ -4,   9,  -5,  -1,  -6],
               [  5,  -4,   6,   9,   6],
               [ -8, -10,   8,   1,   2],
               [  6,  -6,  -5,   5,   4],
               [  0,  -4,   1,  -7,   6],
               [ -2,  -5,  -8,  -6,  -9],
               [ -2,  -8,   3,  -3,   4],
               [  3,   6,   5,   4,   9]])
b = np.array([ 5, -3, -9,  2,  7])

result = np.array([[  84,  -98,   66,  103,  114],
       [  91,  186, -129,   70,  -58],
       [ 111,   31, -158,  -22,  -22]])



def eval_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-10, np.abs(a) * np.abs(b)))
    
    
def numerical_grad_array(f, x, df, h=1e-5):
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    grad = np.zeros_like(x)
    while not it.finished:
        idx = it.multi_index
        
        it[0] += h
        f_pos = f(x)
        
        it[0] -= 2*h
        f_neg = f(x)
        
        it[0] += h
        
        grad[idx] += np.sum((f_pos - f_neg) * df) / (2 * h)
        it.iternext()
    return grad 


def test_linear_forward(f):
    global b,w,x,result
    
    calc_err = 'Ошибка вычислений!\n'
    exec_err = 'Ошибка исполнения, проверьте код на наличие ошибок!'
    inp_s = f'Размеры входных данных: x {x.shape}, W {w.shape}, b {b.shape}.\n'
    out_s = f'Ожидаемый размер выходной матрицы: {result.shape}\n'
    out_r = f'Ожидаемый результат: {result}\n\n'
    try:
        out, cache = f(x, w, b)
    except:
        print(exec_err)
        return
        
    if not out.shape == result.shape:
            print(calc_err + inp_s + out_s + f'Ваш размер выходной матрицы: {out.shape}')
            return
    if not np.abs(out - result).max() == 0:
            print(calc_err + out_r + f'Ваш результат: {out}')
            return
    
    print('Тест пройден успешно!')
    

def test_linear_backward(linear_forward, linear_backward):
    # заполним dout случайными числами
    dout = np.random.rand(16, 10)
    x = np.random.rand(16, 64)
    w = np.random.rand(64, 10)
    b = np.random.rand(10)

    # вычислим значения градиентов при таком dout (аналитически)
    dw, db = linear_backward(dout, [x,w,b]) 
    
    assert w.shape == dw.shape, f'Ожидаемый размер градиента W: {w.shape}, ваш размер: {dw.shape}'
    assert b.shape == db.shape, f'Ожидаемый размер вектора b: {b.shape}, ваш размер: {db.shape}'

    # вычислим численным методом значения градиентов
    dw_ = numerical_grad_array(lambda w: linear_forward(x, w, b)[0], w, dout)
    db_ = numerical_grad_array(lambda b: linear_forward(x, w, b)[0], b, dout)
    

    # посмотрим на ошибки вычислений (значения должны быть меньше 1е-8)
    if eval_error(dw, dw_) > 1e-8:
        print('Градиент матрицы W не совпадает с ожидаемым результатом')
        return

    if eval_error(db, db_) > 1e-8:
        print('Градиент вектора b не совпадает с ожидаемым результатом')
        return

    print('Тест пройден успешно!')
    
    
def test_svm(f):
    l, ds = f(np.array([[1,2,3], [1,2,3]]), [0, 1])
    ds_res = np.array([[-2,  1,  1], [ 0, -1,  1]])

    assert l == 2, 'Значение ошибки вычислено неверно! Убедитесь что вы разделили значение на размер батча!'
    assert np.abs(ds_res - ds).max() == 0, 'Градиент вектора scores вычислен неверно!'
    
    print('Тест пройден успешно!')
    
    
    
        
    
    
    
    
    
    
    
    
    