def prep_roations(x_train, y_train, x_test, y_test):
    new_x_train = []
    new_y_train = []
    new_x_test = []
    new_y_test = []

    for i, num in enumerate(y_train):
        if num not in (6, 9):
            new_y_train.append(num)
            new_x_train.append(x_train[i])
    
    for i, num in enumerate(y_test):
        if num not in (6, 9):
            new_y_test.append(num)
            new_x_test.append(x_test[i])

    return (new_x_train, new_y_train, new_x_test, new_y_test)