import weka.clusterers.ClusterEvaluation;

/**
 * Created by Mostafa on 3/3/2016.
 */
public class SupervisedEvaluater {
    private double TP;
    private double TN;
    private double FP;
    private double FN;
    private double all;
    private double P;
    private double N;
    private double P2;
    private double N2;
    private double precision;
    private double recall;
    private double fScore;
    private double accuracy;
    private double errorRate;
    private double sensitivity;
    private double specificity;

    public SupervisedEvaluater(ClusterEvaluation eval, int index) {
        this.all = eval.numInstances;
        for (int j = 0; j < eval.counts[index].length; j++) {
            P2 += eval.counts[index][j];
        }
        int bestColumn = eval.getClassesToClusters()[index];
        if (bestColumn != -1) {

            this.TP = eval.counts[index][bestColumn];
            this.FP = P2 - TP;
            for (int i = 0; i < eval.counts.length; i++) {
                if (i != index) {
                    this.FN += eval.counts[i][bestColumn];
                }
            }
            P = TP + FN;
            N = all - P;
            N2 = all - P2;
            TN = N - FP;

            this.precision = Double.valueOf((TP / (TP + FP)));
            this.recall = Double.valueOf((TP / (TP + FN)));
            this.fScore = (2 * precision * recall) / (precision + recall);
            this.accuracy = Double.valueOf((TP + TN) / all);
            this.errorRate = 1 - accuracy;
            this.sensitivity = Double.valueOf(TP / P);
            this.specificity = Double.valueOf(TN / N);
        }
    }

    public double getTP() {
        return TP;
    }

    public double getTN() {
        return TN;
    }

    public double getFP() {
        return FP;
    }

    public double getFN() {
        return FN;
    }

    public double getAll() {
        return all;
    }

    public double getP() {
        return P;
    }

    public double getN() {
        return N;
    }

    public double getP2() {
        return P2;
    }

    public double getN2() {
        return N2;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getfScore() {
        return fScore;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getErrorRate() {
        return errorRate;
    }

    public double getSensitivity() {
        return sensitivity;
    }

    public double getSpecificity() {
        return specificity;
    }
}
