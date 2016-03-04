import java.util.*;

/**
 * Created by Mostafa on 2/23/2016.
 */
public class ResultCoefficients {
    private String title;

    private ResultCoefficient cohision = new ResultCoefficient("Cohision(Smaller)", true);
    private ResultCoefficient daviesBouldin = new ResultCoefficient("daviesBouldin(smaller)", true);
    private ResultCoefficient dunn = new ResultCoefficient("dunn(greater)", false);
    private ResultCoefficient silhouette = new ResultCoefficient("silhouette(greater)", false);
    private ResultCoefficient accuracy_macro = new ResultCoefficient("accuracy_macro(greater)", false);
    private ResultCoefficient accuracy_micro = new ResultCoefficient("accuracy_micro(greater)", false);
    private ResultCoefficient precision = new ResultCoefficient("precision(greater)", false);
    private ResultCoefficient recall = new ResultCoefficient("recall(greater)", false);
    private ResultCoefficient fScore = new ResultCoefficient("fScore(greater)", false);
    private ResultCoefficient errorRate = new ResultCoefficient("errorRate(smaller)", true);
    private ResultCoefficient sensitivity = new ResultCoefficient("sensitivity(greater)", false);
    private ResultCoefficient specificity = new ResultCoefficient("specificity(greater)", false);
    List<ResultCoefficients> allResults = new ArrayList<ResultCoefficients>();

    StringBuilder cohisionSB = new StringBuilder();
    StringBuilder daviesBouldinSB = new StringBuilder();
    StringBuilder dunnSB = new StringBuilder();
    StringBuilder silhouetteSB = new StringBuilder();
    StringBuilder accuracy_MacroSB = new StringBuilder();
    StringBuilder accuracy_MicroSB = new StringBuilder();
    StringBuilder precisionSB = new StringBuilder();
    StringBuilder recallSB = new StringBuilder();
    StringBuilder fScoreSB = new StringBuilder();
    StringBuilder errorRateSB = new StringBuilder();
    StringBuilder sensitivitySB = new StringBuilder();
    StringBuilder specificitySB = new StringBuilder();

    public ResultCoefficients(String title) {
        this.title = title;
    }

    public void generateCountResults(StringBuilder sb) {
        sb.append("-------").append("Count result").append("--------").append("\n");
        cohision.generateCountResult(sb);
        daviesBouldin.generateCountResult(sb);
        dunn.generateCountResult(sb);
        silhouette.generateCountResult(sb);
        sb.append("\n");
        accuracy_macro.generateCountResult(sb);
        //accuracy_micro.generateCountResult(sb);
        precision.generateCountResult(sb);
        recall.generateCountResult(sb);
        fScore.generateCountResult(sb);
        errorRate.generateCountResult(sb);
        sensitivity.generateCountResult(sb);
        specificity.generateCountResult(sb);
        sb.append("\n\n");
        sb.toString();
    }

    public void generateResults(StringBuilder sb) {
        sb.append("-------").append(title).append("--------").append("\n");
        cohision.generateResult(sb);
        daviesBouldin.generateResult(sb);
        dunn.generateResult(sb);
        silhouette.generateResult(sb);
        sb.append("\n");
        accuracy_macro.generateResult(sb);
        //accuracy_micro.generateResult(sb);
        precision.generateResult(sb);
        recall.generateResult(sb);
        fScore.generateResult(sb);
        errorRate.generateResult(sb);
        sensitivity.generateResult(sb);
        specificity.generateResult(sb);
        sb.append("\n\n");
        sb.toString();
    }

    public void applyOtherResults(ResultCoefficients another) {
        countAnother(another);
        addResults(another);
        allResults.add(another);
    }

    private void countAnother(ResultCoefficients another) {
        this.cohision.countAnother(another.cohision);
        this.daviesBouldin.countAnother(another.daviesBouldin);
        this.dunn.countAnother(another.dunn);
        this.silhouette.countAnother(another.silhouette);
        this.accuracy_macro.countAnother(another.accuracy_macro);
        this.accuracy_micro.countAnother(another.accuracy_micro);
        this.precision.countAnother(another.precision);
        this.recall.countAnother(another.recall);
        this.fScore.countAnother(another.fScore);
        this.errorRate.countAnother(another.errorRate);
        this.sensitivity.countAnother(another.sensitivity);
        this.specificity.countAnother(another.specificity);
    }

    private void addResults(ResultCoefficients another) {
        addResults(this.cohision, another.cohision.entrySet());
        addResults(this.daviesBouldin, another.daviesBouldin.entrySet());
        addResults(this.dunn, another.dunn.entrySet());
        addResults(this.silhouette, another.silhouette.entrySet());
        addResults(this.accuracy_macro, another.accuracy_macro.entrySet());
        addResults(this.accuracy_micro, another.accuracy_micro.entrySet());
        addResults(this.precision, another.precision.entrySet());
        addResults(this.recall, another.recall.entrySet());
        addResults(this.fScore, another.fScore.entrySet());
        addResults(this.errorRate, another.errorRate.entrySet());
        addResults(this.sensitivity, another.sensitivity.entrySet());
        addResults(this.specificity, another.specificity.entrySet());
    }

    private void addResults(ResultCoefficient coefficient, Set<Map.Entry<String, Double>> entrySet) {
        for (Map.Entry<String, Double> entry : entrySet) {
            if (!coefficient.containsKey(entry.getKey()))
                coefficient.put(entry.getKey(), (double) 0);
            double value = coefficient.get(entry.getKey()) + entry.getValue();
            coefficient.put(entry.getKey(), value);
        }
    }

    public void devideResults(Integer size) {
        devideResults(size, cohision);
        devideResults(size, daviesBouldin);
        devideResults(size, dunn);
        devideResults(size, silhouette);
        devideResults(size, accuracy_macro);
        devideResults(size, accuracy_micro);
        devideResults(size, precision);
        devideResults(size, recall);
        devideResults(size, fScore);
        devideResults(size, errorRate);
        devideResults(size, sensitivity);
        devideResults(size, specificity);
    }

    private void devideResults(Integer size, ResultCoefficient coefficient) {
        for (Map.Entry<String, Double> entry : coefficient.entrySet()) {
            coefficient.put(entry.getKey(), entry.getValue() / size);
        }
    }

    public void printCoefficientsResults(CustomClusters customClusters, StringBuilder sb, String methodName) {
        printResult(cohision, customClusters.computeWithinClusterVariance(), sb, methodName);
//        sb.append("between cluster variance:").append(computeBetweenClusterVariance()).append("\n");
//        sb.append("Fisher:").append(computeFisher()).append("\n");
//        sb.append("total distance:").append(computeTotalMinimumDistance()).append("\n");

        printResult(daviesBouldin, customClusters.computeDaviesBouldin(), sb, methodName);
        printResult(dunn, customClusters.computeDunn(), sb, methodName);
        printResult(silhouette, customClusters.computeSilhouette(), sb, methodName);
        printResult(accuracy_macro, customClusters.computeAccuracy_Macro(), sb, methodName);
//        printResult(accuracy_micro, customClusters.computeAccuracy_Micro(), sb, methodName);
        printResult(precision, customClusters.computePrecision(), sb, methodName);
        printResult(recall, customClusters.computeRecall(), sb, methodName);
        printResult(fScore, customClusters.computeFScore(), sb, methodName);
        printResult(errorRate, customClusters.computeErrorRate(), sb, methodName);
        printResult(sensitivity, customClusters.computeSensitivity(), sb, methodName);
        printResult(specificity, customClusters.computeSpecificity(), sb, methodName);
    }

    private void printResult(ResultCoefficient coefficient, Double result, StringBuilder sb, String methodName) {
        coefficient.put(methodName, result);
        sb.append(coefficient.getTitle()).append(coefficient.get(methodName)).append("\n");
    }

    public Set<String> getKeys() {
        return cohision.getMap().keySet();
    }

    public void generateMatrix_finalize(StringBuilder sb) {
        sb.append(cohisionSB.toString()).append("\n")
                .append(daviesBouldinSB).append("\n")
                .append(dunnSB).append("\n")
                .append(silhouetteSB).append("\n")
                .append(accuracy_MacroSB).append("\n")
//                .append(accuracy_MicroSB).append("\n")
                .append(precisionSB).append("\n")
                .append(recallSB).append("\n")
                .append(fScoreSB).append("\n")
                .append(errorRateSB).append("\n")
                .append(sensitivitySB).append("\n")
                .append(specificitySB).append("\n");
        sb.append("\n");
    }

    public void generateMatrix_header(Set<String> keys) {
        generateHeader(cohisionSB, cohision.getTitle(), keys);
        generateHeader(daviesBouldinSB, daviesBouldin.getTitle(), keys);
        generateHeader(dunnSB, dunn.getTitle(), keys);
        generateHeader(silhouetteSB, silhouette.getTitle(), keys);
        generateHeader(accuracy_MacroSB, accuracy_macro.getTitle(), keys);
//        generateHeader(accuracy_MicroSB, accuracy_micro.getTitle(), keys);
        generateHeader(precisionSB, precision.getTitle(), keys);
        generateHeader(recallSB, recall.getTitle(), keys);
        generateHeader(fScoreSB, fScore.getTitle(), keys);
        generateHeader(errorRateSB, errorRate.getTitle(), keys);
        generateHeader(sensitivitySB, sensitivity.getTitle(), keys);
        generateHeader(specificitySB, specificity.getTitle(), keys);
    }

    public void generateMatrix_row(Set<String> keys, String title) {
        generateRow(keys, cohisionSB, title, cohision);
        generateRow(keys, daviesBouldinSB, title, daviesBouldin);
        generateRow(keys, dunnSB, title, dunn);
        generateRow(keys, silhouetteSB, title, silhouette);
        generateRow(keys, accuracy_MacroSB, title, accuracy_macro);
//        generateRow(keys, accuracy_MicroSB, title, accuracy_micro);
        generateRow(keys, precisionSB, title, precision);
        generateRow(keys, recallSB, title, recall);
        generateRow(keys, fScoreSB, title, fScore);
        generateRow(keys, errorRateSB, title, errorRate);
        generateRow(keys, sensitivitySB, title, sensitivity);
        generateRow(keys, specificitySB, title, specificity);
    }

    private void generateHeader(StringBuilder sb, String methodName, Set<String> keys) {
        sb.append("----").append(methodName).append("---");
        for (String key : keys) {
            sb.append(String.format("%20s", key));
        }
        sb.append("\n");
    }

    private void generateRow(Set<String> keys, StringBuilder cohisionSB, String methodName, ResultCoefficient rc) {
        cohisionSB.append(String.format("%-30s", methodName)).append(":");
        for (String key : keys) {
            cohisionSB.append(String.format("%-20s", rc.get(key))).append(" , ");
        }
        cohisionSB.append("\n");
    }

}
