import os,glob,sys

if __name__ == "__main__":
    didi = {}

    with open("./binding_pdb_input/BindingDB_BindingDB_2D.sdf","r") as F:
        temp_list = []
        for line in F.readlines():
            tline = line.strip()
            """
            if len(temp_list) == 0:
                pass
            else:
                pass"""
            temp_list.append(tline)
            if tline.startswith("$$$$"):
                didi["temp1"] = temp_list
                temp_list = []
                break
            else:
                pass
    print(didi)
    print(os.getcwd())
    for i in didi:
        with open("./temp1.sdf","w") as W:
            for line in didi[i]:
                print(line)
                W.write(line + "\n")



