# 🧮 Matrix Algo Viz - Vizualizare Algoritmi Matrici

[![Streamlit App](https://matrice-vizualizare.streamlit.app/)

O aplicație interactivă și educațională construită cu **Python** și **Streamlit** pentru a vizualiza și compara performanța algoritmilor fundamentali de înmulțire a matricelor.

Proiectul demonstrează diferențele vizuale și de performanță între metoda clasică, metoda pe blocuri și algoritmul Strassen.

## 🌟 Funcționalități Principale

### 1. 🎨 Vizualizare Interactivă (Dashboard)
* **Animație Pas-cu-Pas:** Urmărește execuția algoritmilor în timp real.
* **Highlighting Inteligent:** Evidențiază exact rândurile și coloanele care se înmulțesc la fiecare pas.
* **Formule Matematice:** Afișează formula LaTeX ($C_{ij} = \sum A_{ik} \cdot B_{kj}$) corespunzătoare pasului curent.
* **Control Total:** Butoane de Start, Stop, Reset și slider pentru viteza animației (1-3 secunde/pas).

### 2. ⚡ Benchmark de Performanță
* Compară viteza de execuție reală pe matrici de dimensiuni mari ($N=32$ până la $N=512$).
* Confruntare directă: **Numpy (C/Fortran backend)** vs. **Python Implementations**.
* Grafice comparative și calculul factorului de accelerare (Speedup).

## 🧠 Algoritmi Implementați

| Algoritm | Complexitate | Descriere |
| :--- | :--- | :--- |
| **Standard (Iterativ)** | $O(N^3)$ | Metoda clasică "linie cu coloană". Simplă, dar lentă pentru date mari. |
| **Block (Divide & Conquer)** | $O(N^3)$ | Împarte matricele recursiv în 4 cadrane. Optimizează utilizarea cache-ului procesorului. |
| **Strassen** | $O(N^{2.81})$ | Algoritm avansat care reduce numărul de multiplicări recursive de la 8 la 7. Eficient pentru $N$ foarte mare. |

## 🛠️ Tehnologii Folosite

* **Limbaj:** Python 3.x
* **Interfață:** [Streamlit](https://streamlit.io/)
* **Calcule:** NumPy
* **Vizualizare:** Matplotlib
* **Manipulare Date:** Pandas

## 🚀 Cum rulezi proiectul local

Dacă vrei să rulezi aplicația pe calculatorul tău, urmează acești pași:

1.  **Clonează repozitorul:**
    ```bash
    git clone [https://github.com/username-ul-tau/matrice-vizualizare.git](https://github.com/username-ul-tau/matrice-vizualizare.git)
    cd matrice-vizualizare
    ```

2.  **Instalează dependențele:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pornește aplicația:**
    ```bash
    streamlit run matrix_pro.py
    ```

## 📂 Structura Fișierelor

* `matrix_pro.py` - Codul sursă principal al aplicației.
* `requirements.txt` - Lista librăriilor necesare pentru rulare.
* `README.md` - Documentația proiectului.

---
Proiect realizat pentru a demonstra conceptele de algebră liniară și optimizare algoritmică.
