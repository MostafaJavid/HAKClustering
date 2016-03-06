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
        generateMatrixResults(sb);
        final_sb.generateCountResults(sb);
        if (results.size() == 1)
            final_sb.generateResults(sb);

        return sb.toString();
    }

    public void generateMatrixResults(StringBuilder sb) {
        ResultCoefficients sample = results.get(0).coefficients;
        Set<String> keys = sample.getKeys();
        final_sb.generateMatrix_header(keys);
        for (ResultBeautifier beautifier : results) {
            final_sb.generateMatrix_row(keys, beautifier.getTitle(), beautifier.coefficients);
        }
        final_sb.generateMatrix_finalize(sb);
        sb.toString();
    }

}
