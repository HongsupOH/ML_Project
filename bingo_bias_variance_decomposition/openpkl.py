from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file


def openpkl(file):
    archipelago = load_parallel_archipelago_from_file(file)
    hof = archipelago.hall_of_fame
    print(hof[0])
    print(hof[0].fitness)
    return hof[0]


openpkl('bvd_imp_rep_st_40_p3_p7_2_350.pkl')
    

