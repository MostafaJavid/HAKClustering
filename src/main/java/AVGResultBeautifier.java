import weka.core.pmml.jaxbbindings.Coefficients;

import java.util.*;

/**
 * Created by Mostafa on 2/21/2016.
 */
public class AVGResultBeautifier {
    List<ResultBeautifier> results = new ArrayList<ResultBeautifier>();
    ResultCoefficients final_sb = new ResultCoefficients("average of all results");

    public void add(ResultBeautifier result) {
        this.results.add(result);
    }

    public String printAllResults() {
        StringBuilder sb = new StringBuilder();
        for (ResultBeautifier result : results) {
            sb.append(result.getResult());
            final_sb.applyOtherResults(result.coefficients);
        }
        final_sb.devideResults(results.size());
        generateMatrixResults(sb, results.get(0).coefficients.getCohision().getMap().keySet());
        final_sb.generateCountResults(sb);
        final_sb.generateResults(sb);

        return sb.toString();
    }

    public void generateMatrixResults(StringBuilder sb,Set<String> keys){
        sb.append("-------").append("Matrix").append("--------").append("\n");
        StringBuilder cohisionSB = new StringBuilder();
        StringBuilder daviesBouldinSB = new StringBuilder();
        StringBuilder dunnSB = new StringBuilder();
        StringBuilder silhouetteSB = new StringBuilder();
        ResultCoefficients sample = results.get(0).coefficients;
        generateHeader(cohisionSB,sample.getCohision().getTitle(),keys);
        generateHeader(daviesBouldinSB,sample.getDaviesBouldin().getTitle(),keys);
        generateHeader(dunnSB,sample.getDunn().getTitle(),keys);
        generateHeader(silhouetteSB,sample.getSilhouette().getTitle(),keys);
        for (ResultBeautifier beautifier : results) {
            generateRow(keys, cohisionSB, beautifier.getTitle(), beautifier.coefficients.getCohision());
            generateRow(keys,daviesBouldinSB,beautifier.getTitle(),beautifier.coefficients.getDaviesBouldin());
            generateRow(keys, dunnSB, beautifier.getTitle(), beautifier.coefficients.getDunn());
            generateRow(keys, silhouetteSB, beautifier.getTitle(), beautifier.coefficients.getSilhouette());
        }

        sb.append(cohisionSB.toString()).append("\n")
          .append(daviesBouldinSB).append("\n")
          .append(dunnSB).append("\n")
          .append(silhouetteSB);
        sb.append("\n\n");
        sb.toString();
    }

    private void generateHeader(StringBuilder sb,String methodName,Set<String> keys){
        sb.append("----").append(methodName).append("---");
        for (String key : keys) {
            sb.append(String.format("%20s",key));
        }
        sb.append("\n");
    }

    private void generateRow(Set<String> keys, StringBuilder cohisionSB, String methodName, ResultCoefficient rc) {
        cohisionSB.append(String.format("%-30s",methodName)).append(":");
        for (String key : keys) {
            cohisionSB.append(String.format("%-20s",rc.get(key))).append(" , ");
        }
        cohisionSB.append("\n");
    }


}
