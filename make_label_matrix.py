import os

def make_label_matrix():
    """This method makes a nx3 label matrix from the labels of the dataset"""
    folders = ["dataset\\facebook comments", "dataset\\tweets"]
    for folder_name in folders:
        for file_name in os.listdir(folder_name):
            if 'label.txt'in file_name and file_name[-3:]=='txt':
                label_file = open(folder_name+"\\"+file_name, "r")
                data_file = open(folder_name+"\\"+file_name.rstrip("label.txt")+"data.txt", "r")
                with open(folder_name+"\\"+"Y.txt", "w") as label_matrix:
                    lines = label_file.readlines()
                    X_count = len(data_file.readlines())
                    Y_row = [0.0 for i in xrange(3)]
                    Y_count = 0                    
                    for i in lines:
                        i=i.lstrip().rstrip()
                        if i=="P":
                            Y_row[0]+=1
                        elif i=="N":
                            Y_row[1]+=1
                        elif i=="O":
                            Y_row[2]+=1
                        if sum(Y_row)==1.0:
                            label_matrix.write(" ".join(str(i) for i in Y_row)+"\n")
                            Y_count+=1
                            Y_row = [0.0 for i in xrange(3)]
                        #else skip
                if X_count!=Y_count:
                    print X_count, Y_count
                label_file.close()
                data_file.close()
make_label_matrix()
