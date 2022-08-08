with open("misc/dataset/test.csv",'r') as test:
    with open("misc/dataset/train.csv",'r') as train:
        with open("misc/dataset/dataset.csv",'w') as dataset:
            dataset.writelines(test.readlines())
            dataset.writelines(train.readlines())