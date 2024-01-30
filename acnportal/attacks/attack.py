import random


def fdia_attack(docs, percent_evs_attacked, percent_energy_demanded_change):
    
    total_evs = 0
    
    for d in docs:
        total_evs+=1
    
    number_of_attaced_evs = (total_evs*percent_evs_attacked) // 100
    
    attacked_evs_indexes = random.sample(range(1,total_evs+1), number_of_attaced_evs)    
    
    return attacked_evs_indexes










