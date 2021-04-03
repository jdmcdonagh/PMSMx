class PmsmMeshValues:
    def __init__(self, meshname):
        mshnum = int(meshname.split(".")[0][-1])
        
        if mshnum == 1:
            self.coils = [447, 424, 455, 353, 112, 398]
            self.torqs = [11, 47, 57, 240, 24, 30, 127]

        elif mshnum == 2:
            self.coils = [457, 436, 463, 362, 114, 405]
            self.torqs = [17, 20, 37, 41, 64, 128, 251]

        elif mshnum == 3:
            self.coils = [458, 445, 468, 329, 475, 424]
            self.torqs = [14, 41, 49, 50, 86, 145, 257]

        elif mshnum == 4:
            self.coils = [473, 449, 478, 406, 421, 443]
            self.torqs = [22, 49, 50, 53, 67, 148, 259]

        else:
            print("Mesh Number Not Recognised: ", str(mshnum))