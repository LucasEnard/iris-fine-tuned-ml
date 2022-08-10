with open("misc/dataset/AllCsvGen.txt",'w') as file:
    for i in range(1,101):
        file.writelines(f"DO ##class(community.csvgen).Generate(\"/irisdev/app/misc/dataset/datasets/dataset{i}.txt\",\";\",\"iris.Review\")\n")