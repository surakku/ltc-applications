import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU

import tensorflow as tf

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse


## Mapping for all values
class_map = {
    "Yes": 1,
    "No": 0
}

class_map_null = {
    "Yes": 1,
    "No": 0,
    "": 2,
    '-99': 3
}

statMap = {
    "Alive": 1,
    "Dead": 0,
    "Deceased": 3,
    "": 2
}

respMap = {
    "Normal (30-60)": 0,
    "Tachypnea": 1,
    "": 2,
    "Normal": 3,
    "Tachycardia": 4,
    '-99': 5
}

followMap = {
    "Low": 0,
    "High": 1,
    "": 2,
    '-99': 3
}

genderMap = {
    "Male": 0,
    "Female": 1,
    "Ambiguous": 2,
    "": 3,
    '-99': 4
}

aspMap = {
    "Yes": 0,
    "No": 1,
    "No record": 2,
    "Not available": 3,
    "": 4,
    '-99': 5
}



birthMap = {
    "Home": 0,
    "Institute": 1,
    "": 2,
    '-99': 3
}

autMap = {
    "Yes": 0,
    "No": 1,
    "Not applicable": 2,
    "-": 3,
    "": 4,
    "None": 5,
    '-99': 6
}

defectMap = {
    "Singular": 0,
    "Multiple": 1,
    "": 2,
    '-99': 3
}

btMap = {
    "normal": 0,
    "abnormal": 1,
    "slightly abnormal": 2,
    "inconclusive": 3,
    "": 4,
    '-99': 5
}

disorder_map = {
    "Leigh syndrome":0,
    "Mitochondrial myopathy":1,
    "Cystic fibrosis":2,
    "Tay-Sachs":3,
    "Diabetes":4,
    "Leber's hereditary optic neuropathy": 5,
    "Cancer": 6,
    "Hemochromatosis": 7,
    "Alzheimer's": 8,
    "": 9
    
}



## Transform dataframe of csv into np stack of mapped values
def load_trace(df):
    
    age = df["Patient Age"].fillna(20).values.astype(np.float32)
    
    mGene = df["Genes in mother's side"].values
    mGeneF = np.empty(mGene.shape[0],dtype=np.int32)
    for i in range(mGene.shape[0]):
        mGeneF[i] = class_map_null[mGene[i]]
    
    fGene = df["Inherited from father"].fillna("").values

    fGeneF = np.empty(fGene.shape[0],dtype=np.int32)
    for i in range(fGene.shape[0]):
        fGeneF[i] = class_map_null[str(fGene[i])]
    
    matGene = df["Maternal gene"].fillna("").values
    matGeneF = np.empty(matGene.shape[0],dtype=np.int32)
    for i in range(matGene.shape[0]):
        matGeneF[i] = class_map_null[matGene[i]]
        
    patGene = df["Paternal gene"].fillna("").values
    patGeneF = np.empty(patGene.shape[0],dtype=np.int32)
    for i in range(patGene.shape[0]):
        patGeneF[i] = class_map_null[patGene[i]]
        
    bCell = df["Blood cell count (mcL)"].fillna(20).values.astype(np.float32)
    mAge = df["Mother's age"].fillna(20).values.astype(np.float32)
    fAge = df["Father's age"].fillna(20).values.astype(np.float32)
    
    status = df["Status"].fillna("").values
    statusF = np.empty(status.shape[0],dtype=np.int32)
    for i in range(status.shape[0]):
        statusF[i] = statMap[status[i]]    
    
    respRate = df["Respiratory Rate (breaths/min)"].fillna("").values
    respRateF = np.empty(respRate.shape[0],dtype=np.int32)
    for i in range(respRate.shape[0]):
        respRateF[i] = respMap[respRate[i]] 
        
    hRate = df["Heart Rate (rates/min"].fillna("").values
    hRateF = np.empty(hRate.shape[0],dtype=np.int32)
    for i in range(hRate.shape[0]):
        hRateF[i] = respMap[hRate[i]]
        
    t1 = df["Test 1"].fillna(20).values.astype(np.float32)
    t2 = df["Test 2"].fillna(20).values.astype(np.float32)
    t3 = df["Test 3"].fillna(20).values.astype(np.float32)
    t4 = df["Test 4"].fillna(20).values.astype(np.float32)
    t5 = df["Test 5"].fillna(20).values.astype(np.float32)
    
    followUp = df["Follow-up"].fillna("").values
    followUpF = np.empty(followUp.shape[0],dtype=np.int32)
    for i in range(followUp.shape[0]):
        followUpF[i] = followMap[followUp[i]]
            
    gender = df["Gender"].fillna("").values
    genderF = np.empty(gender.shape[0],dtype=np.int32)
    for i in range(gender.shape[0]):
        genderF[i] = genderMap[gender[i]]
            
    birthAsp = df["Birth asphyxia"].fillna("").values
    birthAspF = np.empty(birthAsp.shape[0],dtype=np.int32)
    for i in range(birthAsp.shape[0]):
        birthAspF[i] = aspMap[birthAsp[i]]
        
    autDefect = df["Autopsy shows birth defect (if applicable)"].fillna("").values
    autDefectF = np.empty(autDefect.shape[0],dtype=np.int32)
    for i in range(autDefect.shape[0]):
        autDefectF[i] = autMap[autDefect[i]]
        
    birthPlace = df["Place of birth"].fillna("").values
    birthPlaceF = np.empty(birthPlace.shape[0],dtype=np.int32)
    for i in range(birthPlace.shape[0]):
        birthPlaceF[i] = birthMap[birthPlace[i]]
        
    folicAcid = df["Folic acid details (peri-conceptional)"].fillna("").values
    folicAcidF = np.empty(folicAcid.shape[0],dtype=np.int32)
    for i in range(folicAcid.shape[0]):
        folicAcidF[i] = class_map_null[folicAcid[i]]
        
    matIll = df["H/O serious maternal illness"].fillna("").values
    matIllF = np.empty(matIll.shape[0],dtype=np.int32)
    for i in range(matIll.shape[0]):
        matIllF[i] = class_map_null[matIll[i]]
        
    radExp = df["H/O radiation exposure (x-ray)"].fillna("").values
    radExpF = np.empty(radExp.shape[0],dtype=np.int32)
    for i in range(radExp.shape[0]):
        radExpF[i] = autMap[radExp[i]]
        
    subAbuse = df["H/O substance abuse"].fillna("").values
    subAbuseF = np.empty(subAbuse.shape[0],dtype=np.int32)
    for i in range(subAbuse.shape[0]):
        subAbuseF[i] = autMap[subAbuse[i]]
        
    assistConception = df["Assisted conception IVF/ART"].fillna("").values
    assistConceptionF = np.empty(assistConception.shape[0],dtype=np.int32)
    for i in range(assistConception.shape[0]):
        assistConceptionF[i] = class_map_null[assistConception[i]]
    
    pregAnom = df["History of anomalies in previous pregnancies"].fillna("").values
    pregAnomF = np.empty(pregAnom.shape[0],dtype=np.int32)
    for i in range(pregAnom.shape[0]):
        pregAnomF[i] = class_map_null[pregAnom[i]]
        
    abort = df["No. of previous abortion"].fillna(20).values.astype(np.float32)
    
    bDefects = df["Birth defects"].fillna("").values
    bDefectsF = np.empty(bDefects.shape[0],dtype=np.int32)
    for i in range(bDefects.shape[0]):
        bDefectsF[i] = defectMap[bDefects[i]]
    
    wbCell = df["White Blood cell count (thousand per microliter)"].fillna(20).values.astype(np.float32)
    
    btResult = df["Blood test result"].fillna("").values
    btResultF = np.empty(btResult.shape[0],dtype=np.int32)
    for i in range(btResult.shape[0]):
        btResultF[i] = btMap[btResult[i]]
        
    symp1 = df["Symptom 1"].fillna(20).values.astype(np.float32)
    symp2 = df["Symptom 2"].fillna(20).values.astype(np.float32)
    symp3 = df["Symptom 3"].fillna(20).values.astype(np.float32)
    symp4 = df["Symptom 4"].fillna(20).values.astype(np.float32)
    symp5 = df["Symptom 5"].fillna(20).values.astype(np.float32)
    
    ## Stack all mapped values
    features = np.stack([age, mGeneF, fGeneF, matGeneF, patGeneF, bCell, mAge, fAge, bCell, statusF, respRateF, hRateF, t1, t2, t3, t4, t5, followUpF, genderF, birthAspF, autDefectF, birthPlaceF, folicAcidF, matIllF, radExpF, subAbuseF, assistConceptionF, pregAnomF, abort, bDefectsF, wbCell, btResultF, symp1, symp2, symp3, symp4, symp5], axis=-1)
    
    ## Map disorders into array
    try:
        disorder = df["Disorder Subclass"].fillna("").values
        disorderF = np.empty(btResult.shape[0],dtype=np.int32)
        for i in range(disorder.shape[0]):
            disorderF[i] = disorder_map[disorder[i]]
    except: 
        print("No disorder feature present")
        disorderF = None

    
    return features, disorderF
    
    


## Cut 2 arrays into matching size sequences
def cut_in_sequences(x, y, seq_len, inc=1):
    

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)






class GeneticData:
    def __init__(self, seq_len=32):
        
        df = pd.read_csv("data/genes/train.csv")
        x, y = load_trace(df)
        
        ## Form training data
        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4)
        print(train_x.shape, train_y.shape)
        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        
        ## Randomize data
        permutation = np.random.RandomState(8302005).permutation(total_seqs)
        print(permutation.shape)
        
        ## First 10% as valid, Next 15% as test, Last 75% as train
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        ## Split into valid, test, and train pairs
        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]   
        
    ## Given batch size, create batches
    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size
        
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)            
        


class GeneModel:
    def __init__(self, model_type, model_size, learning_rate=0.01, model_load=None):
        
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 37])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])
        
        self.model_size = model_size
        head = self.x
        
        ## Set LTC cell parameters
        self.wm = ltc.LTCCell(model_size)
        if model_type.endswith("_rk"):
            self.wm._solver = ltc.ODESolver.RungeKutta
        elif model_type.endswith("_ex"):
            self.wm._solver = ltc.ODESolver.Explicit
        else:
            self.wm._solver = ltc.ODESolver.SemiImplicit
        
        ## Create RNN model using LTC Cell
        head, _ = tf.nn.dynamic_rnn(
            self.wm, head, dtype=tf.float32, time_major=True
        )
        self.constrain_op = self.wm.get_param_constrain_op()
        
        ## Run through once to get initial loss with random weights
        target_y = tf.expand_dims(self.target_y, axis=-1)
        self.y = tf.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        
        ## Get shape and inital loss, set the optimizer
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)
        
        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        ## Create model session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        ## I.O Boilerplate
        self.result_file = os.path.join(
            "results", "genes", "{}_{}.csv".format(model_type, model_size)
        )
        if not os.path.exists("results/genes"):
            os.makedirs("results/genes")
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n"
                )
                
        self.checkpoint_path = os.path.join(
            "tf_sessions", "genes", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/genes"):
            os.makedirs("tf_sessions/genes")

        self.saver = tf.train.Saver()
        
    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)
        print("Restored: " + self.checkpoint_path)
        
    def predict(self, data):
        return self.sess.run(self.y, {self.x: data})
    
    def run(self, input_data):
        return self.sess.run(self.y, {self.x: input_data})
        
    def fit(self, gene_data, epochs, verbose=True, log_period=50):
        
        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            ## Train Iteratively
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gene_data.test_x, self.target_y: gene_data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gene_data.valid_x, self.target_y: gene_data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                    self.save()
            losses = []
            accs = []
            for batch_x, batch_y in gene_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y}
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)
                
                losses.append(loss)
                accs.append(acc)
                
            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))):
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        with open(self.result_file, "a") as f:
            f.write(
                "{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                    best_epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )
            
        
            
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ltc")
    parser.add_argument("--log", default=1, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()
    
    gene_data = GeneticData()
    model = GeneModel(model_type=args.model, model_size=args.size)
    
    model.fit(gene_data, epochs=args.epochs, log_period=args.log)
    
    
    
    
    

