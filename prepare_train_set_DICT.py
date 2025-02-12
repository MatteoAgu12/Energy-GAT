import pickle
from pymatgen.core import Structure
from typing import List, Dict
import numpy as np
from scipy.interpolate import CubicSpline

def divide_in_bands(energies: List, vbands: int, cbands: int) -> List[Dict]:
    cond_bands = {}
    val_bands = {}
    
    for i in range(vbands):
        val_bands["Valence" + str(i+1)] = []
        
    for i in range(cbands):
        cond_bands["Conduction" + str(i+1)] = []
        
    for j, e in enumerate(energies):
        rest = j % (vbands + cbands)
        if rest < vbands:
            val_bands["Valence" + str(rest + 1)].append(e)
        elif rest < (vbands + cbands):
            cond_bands["Conduction" + str(rest + 1 - vbands)].append(e)
    
    return [val_bands, cond_bands]

def interpolate(x_vals, y_vals, points=50):
    if len(x_vals) != len(y_vals):
        raise ValueError("The two input lists must have the same length.")
    
    spline = CubicSpline(x_vals, y_vals)
    x_interp = np.linspace(min(x_vals), max(x_vals), points)
    y_interp = spline(x_interp)
    
    return y_interp

def get_GW_gap(DFT_Vband, DFT_Cband, GW_Vband, GW_Cband, DFT_gap):
    correction = 0
    valence = []
    conduction = []
    
    for i in range(len(DFT_Cband)):
        valence.append(DFT_Vband[i] + GW_Vband[i])
        conduction.append(DFT_Cband[i] + GW_Cband[i])
    
    max_V = max(valence)
    min_C = min(conduction)
    real_gap = min_C - max_V
    correction = real_gap - DFT_gap
    
    return correction

def create_DB_interpol(start: int = 0, end: int = -1) -> List[Dict]:
    loaded_data = None
    
    with open(r"C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\df_parsed_DB_full_20241127.pkl", "rb") as file:
        loaded_data = pickle.load(file)[start:end]
        
    DB = []   
    
    for material in loaded_data:        
        out = {}

        out["vbands"] = material['vbands_included']
        out["cbands"] = material["cbands_included"]
        
        valence_title = 'Valence' + str(out['vbands']) 
        
        linearized_bands_DFT = material["DFT_bands_windowed_linearized"]
        DFT_bands = divide_in_bands(linearized_bands_DFT, out['vbands'], out['cbands'])
        linearized_corrections_GW = material["GW_QPc_windowed_linearized"]
        GW_corrections = divide_in_bands(linearized_corrections_GW, out['vbands'], out['cbands'])
        
        out["gap_DFT"] = material["gap_DFT"]
        out['GW_correction'] = get_GW_gap(DFT_bands[0][valence_title], DFT_bands[1]['Conduction1'], 
                                                         GW_corrections[0][valence_title], GW_corrections[1]['Conduction1'], 
                                                         out["gap_DFT"])

        VB = interpolate(np.linspace(0, 16, len(DFT_bands[0][valence_title])), DFT_bands[0][valence_title])
        CB = interpolate(np.linspace(0, 16, len(DFT_bands[1]['Conduction1'])), DFT_bands[1]['Conduction1'])
        out['DFT_valence_band'] = VB
        out['DFT_conduction_band'] = CB 
    
        structure: Structure = material["structure"]
        structure = structure.as_dict()
    
        a = structure['lattice']['a']
        b = structure['lattice']['b']
        c = structure['lattice']['c']
    
        out["A"] = np.array(structure['lattice']['matrix'][0]) * a
        out["B"] = np.array(structure['lattice']['matrix'][1]) * b
        out["C"] = np.array(structure['lattice']['matrix'][2]) * c
    
        out["alpha"] = structure['lattice']['alpha']
        out["beta"] = structure['lattice']['beta']
        out["gamma"] = structure['lattice']['gamma']
        out["volume"] = structure['lattice']['volume']
    
        out['atoms'] = {}
        sites = structure['sites']
        for site in sites:
            species = site['species']
        
            for specie in species:
                element = str(specie['element'])
                keys = out['atoms'].keys()
                if not element in keys:
                    out['atoms'][element] = []
                    out['atoms'][element].append(site['xyz'])
                else:
                    out['atoms'][element].append(site['xyz'])
                    
        DB.append(out)
        
    return DB

def create_DB_K(start: int = 0, end: int = -1) -> List[Dict]:
    loaded_data = None
    
    with open(r"C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\df_parsed_DB_full_20241127.pkl", "rb") as file:
        loaded_data = pickle.load(file)[start:end]
        
    DB = []   
    
    for material in loaded_data:        
        out = {}

        out["vbands"] = material['vbands_included']
        out["cbands"] = material["cbands_included"]
        
        valence_title = 'Valence' + str(out['vbands']) 
        
        linearized_bands_DFT = material["DFT_bands_windowed_linearized"]
        DFT_bands = divide_in_bands(linearized_bands_DFT, out['vbands'], out['cbands'])
        linearized_corrections_GW = material["GW_QPc_windowed_linearized"]
        GW_corrections = divide_in_bands(linearized_corrections_GW, out['vbands'], out['cbands'])
        
        out["gap_DFT"] = material["gap_DFT"]
        out['GW_correction'] = get_GW_gap(DFT_bands[0][valence_title], DFT_bands[1]['Conduction1'], 
                                                         GW_corrections[0][valence_title], GW_corrections[1]['Conduction1'], 
                                                         out["gap_DFT"])

        #VB = interpolate(np.linspace(0, 16, len(DFT_bands[0][valence_title])), DFT_bands[0][valence_title])
        #CB = interpolate(np.linspace(0, 16, len(DFT_bands[1]['Conduction1'])), DFT_bands[1]['Conduction1'])
        VB = DFT_bands[0][valence_title]
        CB = DFT_bands[1]['Conduction1']
        
        while len(VB) < 112:
            VB.append(0)
        while len(CB) < 112:
            CB.append(0)
        if len(VB) != len(CB) or len(VB) != 112: raise ValueError(f'Something wrong with the size of CB and VB. {len(VB)}')
        
        out['DFT_valence_band'] = VB
        out['DFT_conduction_band'] = CB 
        
        k_points = []        
        for point in material['actual_kpoints']:
            k_points.append(point[0])
            k_points.append(point[1])
            k_points.append(point[2])
            
        while len(k_points) != 112*3:
            k_points.append(0)
            
        if len(k_points) != 3 * len(VB): raise ValueError('Error in the number of k-points.')
        
        out['k_points'] = k_points
    
        structure: Structure = material["structure"]
        structure = structure.as_dict()
    
        a = structure['lattice']['a']
        b = structure['lattice']['b']
        c = structure['lattice']['c']
    
        out["A"] = np.array(structure['lattice']['matrix'][0]) * a
        out["B"] = np.array(structure['lattice']['matrix'][1]) * b
        out["C"] = np.array(structure['lattice']['matrix'][2]) * c
    
        out["alpha"] = structure['lattice']['alpha']
        out["beta"] = structure['lattice']['beta']
        out["gamma"] = structure['lattice']['gamma']
        out["volume"] = structure['lattice']['volume']
    
        out['atoms'] = {}
        sites = structure['sites']
        for site in sites:
            species = site['species']
        
            for specie in species:
                element = str(specie['element'])
                keys = out['atoms'].keys()
                if not element in keys:
                    out['atoms'][element] = []
                    out['atoms'][element].append(site['xyz'])
                else:
                    out['atoms'][element].append(site['xyz'])
                    
        DB.append(out)
        
    return DB

def save_dict_to_pkl(filename: str, DB: List[Dict]) -> None:
    with open(filename, "wb") as file:
        pickle.dump(DB, file)
    
    print("Dictionary saved in the pkl file!")


if __name__ == '__main__':
    DB_interpol = create_DB_interpol()
    DB_K = create_DB_K()
    save_dict_to_pkl(r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_interpol.pkl', DB_interpol)
    save_dict_to_pkl(r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_K.pkl', DB_K)
