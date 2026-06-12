#!/usr/bin/env python3
"""Build English copy of Report_CTG_ITALIAN.docx preserving layout and images."""

from __future__ import annotations

import html
import re
import shutil
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "Report_CTG_ITALIAN.docx"
TARGET_DOCX = BASE / "Report_CTG_ENGLISH.docx"
TARGET_PDF = BASE / "Report_CTG_ENGLISH.pdf"

# Longest strings first to avoid partial replacements.
TRANSLATIONS: list[tuple[str, str]] = [
    (
        "Il progetto consiste in un esperimento di classificazione in Python. Ho scelto di occuparmi di un dataset medico che contiene misurazioni estratte da cardiotocogrammi fetali, cioè quei tracciati che si fanno in gravidanza per monitorare il battito del feto e le contrazioni uterine.",
        "This project is a classification experiment in Python. I worked with a medical dataset containing measurements extracted from fetal cardiotocograms—the traces recorded during pregnancy to monitor fetal heart rate and uterine contractions.",
    ),
    (
        "Ho scelto questo dataset per diversi motivi. Innanzitutto ha 2.126 campioni e 21 feature numeriche, una dimensione che si presta bene sia agli algoritmi classici sia al deep learning. Il fatto di avere 3 classi (normal, suspect, pathologic) lo rende più interessante di un semplice binario. Infine, trattandosi di medicina, la spiegabilità diventa particolarmente importante: aiuta a capire quali variabili pesano di più nelle decisioni del modello.",
        "I chose this dataset for several reasons. First, it has 2,126 samples and 21 numerical features—a size that suits both classical algorithms and deep learning. Having 3 classes (normal, suspect, pathologic) makes it more interesting than a simple binary problem. Finally, in a medical context, explainability is especially important: it helps identify which variables matter most in the model's decisions.",
    ),
    (
        "Oltre alla PCA, ai tre classificatori, alle metriche di valutazione e alle visualizzazioni, ho scelto di includere sia l’approccio deep learning sia l’analisi di spiegabilità con SHAP.",
        "Beyond PCA, the three classifiers, evaluation metrics, and visualizations, I also included both a deep learning approach and SHAP-based explainability analysis.",
    ),
    (
        "Il dataset contiene 21 feature misurate automaticamente dai tracciati CTG. Tra queste ci sono: LB (frequenza cardiaca fetale di base), AC (accelerazioni al secondo), FM (movimenti fetali), UC (contrazioni uterine), diverse tipologie di decelerazioni (DL, DS, DP), misure di variabilità a breve e lungo termine (ASTV, MSTV, ALTV, MLTV), e statistiche dell'istogramma della frequenza cardiaca (Width, Min, Max, Mode, Mean, Median, Variance, ecc.).",
        "The dataset contains 21 features automatically measured from CTG traces. These include: LB (fetal heart rate baseline), AC (accelerations per second), FM (fetal movements), UC (uterine contractions), several types of decelerations (DL, DS, DP), short- and long-term variability measures (ASTV, MSTV, ALTV, MLTV), and fetal heart rate histogram statistics (Width, Min, Max, Mode, Mean, Median, Variance, etc.).",
    ),
    (
        "La variabile target NSP classifica lo stato fetale in tre classi: Normal (1.655 campioni, il 77.8%), Suspect (295, il 13.9%) e Pathologic (176, l'8.3%). Come si vede, il dataset è sbilanciato: la stragrande maggioranza dei casi è normale, il che riflette la realtà clinica ma rende la classificazione delle classi minoritarie più impegnativa.",
        "The target variable NSP classifies fetal state into three classes: Normal (1,655 samples, 77.8%), Suspect (295, 13.9%), and Pathologic (176, 8.3%). As expected, the dataset is imbalanced: the vast majority of cases are normal, which reflects clinical reality but makes classifying minority classes more challenging.",
    ),
    (
        "Un aspetto che ho notato subito guardando la matrice di correlazione è che molte feature sono fortemente correlate tra loro: Mode, Mean e Median, ad esempio, hanno correlazioni superiori a 0.90. Questo ci anticipa che la PCA sarà molto efficace nel ridurre la dimensionalità.",
        "One thing I noticed immediately from the correlation matrix is that many features are strongly correlated: Mode, Mean, and Median, for example, have correlations above 0.90. This suggests that PCA will be very effective at reducing dimensionality.",
    ),
    (
        "Ho diviso i dati in tre set: training (70%), validation (20%) e test (10%), usando la stratificazione per mantenere le proporzioni delle classi in ciascun set. Ho usato la stessa suddivisione per tutti gli esperimenti (shallow, deep learning e SHAP) così i risultati sono direttamente confrontabili.",
        "I split the data into three sets: training (70%), validation (20%), and test (10%), using stratification to preserve class proportions in each set. I used the same split for all experiments (shallow, deep learning, and SHAP) so the results are directly comparable.",
    ),
    (
        "Ho applicato lo StandardScaler, che normalizza ogni feature sottraendo la media e dividendo per la deviazione standard. Questo passaggio è fondamentale per la PCA (che altrimenti sarebbe influenzata dalle feature con valori più grandi) e per gli algoritmi sensibili alla scala come SVM e Logistic Regression. Il fit dello scaler l'ho fatto sul training set e poi ho trasformato validation e test con gli stessi parametri.",
        "I applied StandardScaler, which normalizes each feature by subtracting the mean and dividing by the standard deviation. This step is essential for PCA (which would otherwise be dominated by features with larger values) and for scale-sensitive algorithms such as SVM and Logistic Regression. I fit the scaler on the training set and then transformed validation and test with the same parameters.",
    ),
    (
        "Ho applicato la PCA alle feature standardizzate del training set. Analizzando la varianza, ho visto che bastano 14 componenti per conservare il 95% dell'informazione originale. Questa scelta riduce la dimensione del problema eliminando le ridondanze dovute alle forti correlazioni tra le 21 feature di partenza.",
        "I applied PCA to the standardized training features. Analyzing the variance, I found that 14 components are enough to retain 95% of the original information. This reduces the problem size by removing redundancy due to strong correlations among the 21 original features.",
    ),
    (
        "Ho anche fatto una proiezione a 2 componenti per visualizzare i dati. Lo scatter plot mostra che la classe Pathologic tende a formare un cluster parzialmente separato, mentre Normal e Suspect hanno vaste zone in cui si sovrappongono. Questo ha subito segnalato che distinguere Normal da Pathologic sarebbe stato più semplice rispetto a distinguere Normal da Suspect.",
        "I also created a 2-component projection to visualize the data. The scatter plot shows that the Pathologic class tends to form a partially separate cluster, while Normal and Suspect overlap extensively. This immediately suggested that separating Normal from Pathologic would be easier than separating Normal from Suspect.",
    ),
    (
        "Ho poi usato le 14 componenti PCA come input per tutti i classificatori successivi.",
        "I then used the 14 PCA components as input for all subsequent classifiers.",
    ),
    (
        "Ho implementato tre classificatori, scelti perché rappresentano approcci diversi al problema:",
        "I implemented three classifiers, chosen because they represent different approaches to the problem:",
    ),
    (
        "Logistic Regression: L'ho usata come baseline lineare. Nonostante la sua semplicità, funziona discretamente perché la PCA ha già “raddrizzato” i dati lungo le direzioni di massima varianza.",
        "Logistic Regression: I used it as a linear baseline. Despite its simplicity, it performs reasonably well because PCA has already “straightened” the data along the directions of maximum variance.",
    ),
    (
        "SVM con kernel RBF: Ho scelto il kernel RBF perché introduce non-linearità e permette al modello di tracciare confini decisionali curvi. Ho impostato probability=True per poter calcolare le probabilità e poter valutare il modello tramite la curva ROC AUC, che in ambito clinico è essenziale per capire quanto il sistema sia affidabile.",
        "SVM with RBF kernel: I chose the RBF kernel because it introduces non-linearity and allows the model to draw curved decision boundaries. I set probability=True to compute probabilities and evaluate the model via ROC AUC, which is essential in clinical settings to assess system reliability.",
    ),
    (
        "Random Forest: Un ensemble di 200 alberi decisionali con profondità massima 10. L'ho scelto perché è robusto, gestisce bene i dataset sbilanciati e si adatta bene a SHAP (TreeExplainer).",
        "Random Forest: An ensemble of 200 decision trees with maximum depth 10. I chose it because it is robust, handles imbalanced datasets well, and works well with SHAP (TreeExplainer).",
    ),
    (
        " L'ho usata come baseline lineare. Nonostante la sua semplicità, funziona discretamente perché la PCA ha già “raddrizzato” i dati lungo le direzioni di massima varianza.",
        " I used it as a linear baseline. Despite its simplicity, it performs reasonably well because PCA has already “straightened” the data along the directions of maximum variance.",
    ),
    (
        " Ho scelto il kernel RBF perché introduce non-linearità e permette al modello di tracciare confini decisionali curvi. Ho impostato probability=True per poter calcolare le probabilità e poter valutare il modello tramite la curva ROC AUC, che in ambito clinico è essenziale per capire quanto il sistema sia affidabile.",
        " I chose the RBF kernel because it introduces non-linearity and allows the model to draw curved decision boundaries. I set probability=True to compute probabilities and evaluate the model via ROC AUC, which is essential in clinical settings to assess system reliability.",
    ),
    (
        " Un ensemble di 200 alberi decisionali con profondità massima 10. L'ho scelto perché è robusto, gestisce bene i dataset sbilanciati e si adatta bene a SHAP (TreeExplainer).",
        " An ensemble of 200 decision trees with maximum depth 10. I chose it because it is robust, handles imbalanced datasets well, and works well with SHAP (TreeExplainer).",
    ),
    (
        "Ho calcolato le metriche sul test set (213 campioni). Precision, Recall e F1-Score sono calcolati con media macro, che dà lo stesso peso a tutte le classi indipendentemente dalla loro frequenza. La ROC AUC è calcolata con strategia One-vs-Rest.",
        "I computed metrics on the test set (213 samples). Precision, Recall, and F1-Score use macro averaging, giving equal weight to all classes regardless of frequency. ROC AUC is computed with a One-vs-Rest strategy.",
    ),
    (
        "Il Random Forest è il migliore tra i modelli shallow su quasi tutte le metriche. Il DNN mostra un lieve vantaggio numerico in F1-score e ROC AUC, ma il margine rispetto a Random Forest è ridotto e va interpretato con cautela. La SVM ha la precision più alta dopo RF ma un recall più basso: in pratica è più “prudente”, quando predice Pathologic di solito ha ragione, ma ne manca di più tra le classi minoritarie.",
        "Random Forest is the best among shallow models on almost all metrics. The DNN shows a slight numerical advantage in F1-score and ROC AUC, but the margin over Random Forest is small and should be interpreted cautiously. SVM has the highest precision after RF but lower recall: in practice it is more “conservative”—when it predicts Pathologic it is usually correct, but it misses more cases among minority classes.",
    ),
    (
        "Il Random Forest è il migliore tra i modelli shallow su quasi tutte le metriche. Il DNN mostra un lieve vantaggio numerico in F1-score e ROC AUC, ma il margine rispetto a Random Forest è ridotto e va interpretato con cautela. La SVM ha la precision più alta dopo RF ma un recall più basso: in pratica è più “prudente”, quando predice Pathologic di solito ha ragione, ma ne manca di più ",
        "Random Forest is the best among shallow models on almost all metrics. The DNN shows a slight numerical advantage in F1-score and ROC AUC, but the margin over Random Forest is small and should be interpreted cautiously. SVM has the highest precision after RF but lower recall: in practice it is more “conservative”—when it predicts Pathologic it is usually correct, but it misses more ",
    ),
    (
        "ne manca di più tra le classi minoritarie.",
        "cases among minority classes.",
    ),
    (
        "Le confusion matrix mostrano un pattern comune a tutti i modelli: i casi Normal vengono classificati molto bene (oltre il 95%), ma la classe Suspect viene spesso confusa con Normal. Questo ha senso clinicamente: i casi sospetti sono per definizione al confine.",
        "The confusion matrices show a common pattern across all models: Normal cases are classified very well (over 95%), but the Suspect class is often confused with Normal. This makes clinical sense: suspect cases are by definition borderline.",
    ),
    (
        "Le curve ROC confermano che tutti i modelli hanno eccellenti capacità discriminative, con AUC sopra 0.95 per tutte le classi. La classe Normal è la più facile da identificare, la Pathologic ha AUC leggermente inferiore ma comunque molto alto.",
        "The ROC curves confirm that all models have excellent discriminative ability, with AUC above 0.95 for all classes. Normal is the easiest class to identify; Pathologic has slightly lower AUC but still very high.",
    ),
    (
        "Ho implementato una rete neurale fully connected con TensorFlow/Keras. L'architettura che ho scelto è:",
        "I implemented a fully connected neural network with TensorFlow/Keras. The architecture I chose is:",
    ),
    (
        "Ho usato i layer di Dropout per regolarizzare e limitare l'overfitting, dato che il dataset non è molto grande. L'ottimizzatore è Adam con categorical cross-entropy come loss. Ho impostato EarlyStopping sulla validation loss, così l'addestramento si ferma quando la generalizzazione smette di migliorare.",
        "I used Dropout layers to regularize and limit overfitting, since the dataset is not very large. The optimizer is Adam with categorical cross-entropy as the loss. I set EarlyStopping on validation loss so training stops when generalization stops improving.",
    ),
    (
        "L’uso della PCA è stato mantenuto per coerenza della pipeline e finalità didattiche; non implica ottimalità per tutti i modelli, in particolare per gli ensemble ad alberi.",
        "PCA was kept for pipeline consistency and educational purposes; this does not imply optimality for all models, especially tree-based ensembles.",
    ),
    (
        "Sul test set il modello Deep Learning raggiunge Accuracy 0.9108, F1-Score macro 0.8477 e ROC AUC 0.9729: nel confronto sul test set risulta leggermente superiore per F1 macro e ROC AUC, ma le prestazioni restano molto vicine a quelle del Random Forest, il quale resta comunque molto competitivo e più semplice da interpretare con le feature importance.",
        "On the test set, the Deep Learning model reaches Accuracy 0.9108, macro F1-Score 0.8477, and ROC AUC 0.9729: in the test-set comparison it is slightly ahead on macro F1 and ROC AUC, but performance remains very close to Random Forest, which is still highly competitive and easier to interpret via feature importance.",
    ),
    (
        "Guardando la tabella e il grafico di confronto, i quattro modelli hanno prestazioni vicine, ma non identiche. Nel confronto sul test set, Deep Learning (DNN) e Random Forest mostrano prestazioni complessivamente comparabili; il DNN ha un vantaggio numerico lieve su F1 macro e ROC AUC.",
        "Looking at the comparison table and chart, the four models have similar but not identical performance. On the test set, Deep Learning (DNN) and Random Forest show broadly comparable results; the DNN has a slight numerical edge on macro F1 and ROC AUC.",
    ),
    (
        "Un aspetto comune a tutti i modelli è la difficoltà sulla classe Suspect, spesso confusa con Normal: questo non dipende solo dall’algoritmo scelto, ma dalla sovrapposizione tra profili “normali” e “sospetti” nel dataset.",
        "A common issue across all models is difficulty with the Suspect class, often confused with Normal: this depends not only on the chosen algorithm, but on overlap between “normal” and “suspicious” profiles in the dataset.",
    ),
    (
        "Questo ci dice una cosa importante: il collo di bottiglia non è l'algoritmo, ma la natura dei dati stessi. Normal e Suspect hanno profili clinici che si sovrappongono, e nessun modello, per quanto sofisticato, riesce a separarli completamente.",
        "This highlights an important point: the bottleneck is not the algorithm, but the data itself. Normal and Suspect have overlapping clinical profiles, and no model, however sophisticated, can separate them completely.",
    ),
    (
        "Per la spiegabilità ho usato due livelli, entrambi sul Random Forest addestrato sulle feature originali standardizzate (senza PCA), così le variabili restano interpretabili (es. ASTV, MSTV, AC) e non come componenti principali astratte.",
        "For explainability I used two levels, both on the Random Forest trained on standardized original features (without PCA), so variables remain interpretable (e.g. ASTV, MSTV, AC) rather than abstract principal components.",
    ),
    (
        "Prima ho analizzato le feature importance del Random Forest (feature_importances_). In questo run le variabili con peso maggiore risultano soprattutto MSTV, ASTV, ALTV, Mean e Median (ordine basato sulle importanze RF).",
        "First, I analyzed Random Forest feature importance (feature_importances_). In this run, the highest-weight variables are mainly MSTV, ASTV, ALTV, Mean, and Median (ordered by RF importances).",
    ),
    (
        "Poi ho applicato SHAP con TreeExplainer sullo stesso Random Forest. SHAP può dare enfasi leggermente diverse rispetto alle importanze medie, perché spiega il contributo delle variabili in modo più “locale” e può cambiare a seconda della classe predetta.",
        "Then I applied SHAP with TreeExplainer on the same Random Forest. SHAP can emphasize variables slightly differently from mean importances, because it explains contributions more “locally” and can vary by predicted class.",
    ),
    (
        "Complessivamente, variabilità a breve/lungo termine (es. MSTV/ASTV/ALTV) e segnali legati a dinamica della frequenza (es. correlati ad accelerazioni e pattern dell'istogramma) sono tra i fattori più rilevanti emersi dall'analisi. Questo non sostituisce il giudizio clinico, ma aiuta a capire quali variabili il modello sta usando con più peso nelle decisioni.",
        "Overall, short/long-term variability (e.g. MSTV/ASTV/ALTV) and signals related to heart-rate dynamics (e.g. linked to accelerations and histogram patterns) are among the most relevant factors from the analysis. This does not replace clinical judgment, but helps clarify which variables the model weights most in its decisions.",
    ),
    (
        "Questo progetto mi ha permesso di vedere come si costruisce una pipeline di machine learning completa, dal preprocessing all'interpretazione dei risultati. Alcuni aspetti che mi hanno colpito:",
        "This project showed me how to build a complete machine learning pipeline, from preprocessing to interpreting results. A few points that stood out:",
    ),
    (
        "La PCA conferma molta ridondanza tra le feature: da 21 feature originali a 14 componenti per il 95% di varianza, in linea con le correlazioni osservate (es. Mode/Mean/Median).",
        "PCA confirms substantial redundancy among features: from 21 original features to 14 components for 95% variance, consistent with observed correlations (e.g. Mode/Mean/Median).",
    ),
    (
        "Nel confronto sul test set, DNN e Random Forest ottengono risultati molto vicini: il DNN ha un lieve vantaggio in F1 macro (0.848) e ROC AUC (0.973), mentre il Random Forest resta il migliore tra gli shallow e più immediato da interpretare.",
        "On the test set, DNN and Random Forest achieve very similar results: the DNN has a slight edge in macro F1 (0.848) and ROC AUC (0.973), while Random Forest remains the best among shallow models and easier to interpret.",
    ),
    (
        "La parte di spiegabilità è utile soprattutto se letta a due livelli: importanze RF (semplici) e SHAP (più dettagliato). Questo rende più chiaro cosa guida le predizioni.",
        "The explainability section is most useful when read at two levels: RF importances (simple) and SHAP (more detailed). This clarifies what drives predictions.",
    ),
    (
        "Tra i miglioramenti futuri prioritari: gestione esplicita dello sbilanciamento (es. class_weight), validazione più robusta (cross-validation) e tuning sistematico degli iperparametri.",
        "Priority future improvements: explicit handling of class imbalance (e.g. class_weight), more robust validation (cross-validation), and systematic hyperparameter tuning.",
    ),
    (
        "Limitazioni: il test set contiene 213 campioni, di cui 18 Pathologic; per questa classe le metriche possono variare sensibilmente anche con poche predizioni diverse. I risultati vanno quindi letti come indicativi sul singolo split, in assenza di cross-validation.",
        "Limitations: the test set contains 213 samples, including 18 Pathologic; for this class, metrics can shift noticeably with only a few different predictions. Results should therefore be read as indicative for this single split, without cross-validation.",
    ),
    ("Progetto di esame", "Exam project"),
    ("Classificazione dello stato fetale", "Fetal state classification"),
    ("Studente: Claudio Scamporlino", "Student: Claudio Scamporlino"),
    ("Matricola: [redacted for publication]", "Student ID: [redacted for publication]"),
    ("Esame: Apprendimento Automatico e Apprendimento Profondo (UTSR)", "Course: Machine Learning and Deep Learning (UTSR)"),
    ("Docente: Prof.ssa Noemi Scarpato", "Instructor: Prof. Noemi Scarpato"),
    ("Progetto: Classificazione dello stato fetale ", "Project: Fetal state classification "),
    ("Progetto: Classificazione dello stato fetale", "Project: Fetal state classification"),
    ("1. Introduzione", "1. Introduction"),
    ("2. Il dataset", "2. The dataset"),
    ("3. Preprocessing", "3. Preprocessing"),
    ("4. Task 1 - Analisi PCA", "4. Task 1 - PCA analysis"),
    ("5. Task 2 - Algoritmi di classificazione", "5. Task 2 - Classification algorithms"),
    ("6. Task 3 - Metriche di valutazione", "6. Task 3 - Evaluation metrics"),
    ("7. Task 4 - Confusion Matrix e ROC Curves", "7. Task 4 - Confusion matrices and ROC curves"),
    ("8. Task 5a - Deep Learning", "8. Task 5a - Deep Learning"),
    ("9. Confronto tra i modelli", "9. Model comparison"),
    ("10. Task 5b - Spiegabilità", "10. Task 5b - Explainability"),
    ("nclusioni", "nclusions"),
    ("Appendice — Codice Python", "Appendix — Python code"),
    ("Di seguito il codice completo del notebook Python, suddiviso per celle.", "Below is the complete Python notebook code, organized by cells."),
    (">Modello", ">Model"),
    ("Modello", "Model"),
]


def translate_xml(xml: str) -> str:
    for src, dst in TRANSLATIONS:
        xml = xml.replace(src, dst)
        # Also handle XML-escaped apostrophes in source docx.
        xml = xml.replace(src.replace("'", "&apos;"), dst.replace("'", "&apos;"))
        xml = xml.replace(src.replace("'", "&#39;"), dst.replace("'", "&#39;"))
    return xml


def build_docx() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source report: {SOURCE}")

    shutil.copy2(SOURCE, TARGET_DOCX)

    with zipfile.ZipFile(TARGET_DOCX, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    entries["word/document.xml"] = translate_xml(entries["word/document.xml"].decode("utf-8")).encode("utf-8")

    with zipfile.ZipFile(TARGET_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)

    print(f"Created {TARGET_DOCX.name}")


def verify_no_italian_left() -> None:
    italian_markers = [
        "Introduzione",
        "Il dataset",
        "Spiegabilità",
        "Conclusioni",
        "Progetto di esame",
        "Classificazione dello stato fetale",
        "Algoritmi di classificazione",
        "Metriche di valutazione",
        "Confronto tra i modelli",
        "Di seguito il codice",
        "Modello",
        "Studente:",
        "Matricola:",
        "Docente:",
        "Esame:",
    ]
    with zipfile.ZipFile(TARGET_DOCX) as z:
        text = z.read("word/document.xml").decode("utf-8")
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))
    leftovers = [m for m in italian_markers if m in plain]
    if leftovers:
        raise RuntimeError(f"Italian markers still present: {leftovers}")
    print("Verification passed: no obvious Italian markers left.")


if __name__ == "__main__":
    build_docx()
    verify_no_italian_left()
