import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Mostafa on 2/21/2016.
 */
public class AVGResultBeautifier {
    List<ResultBeautifier> results = new ArrayList<ResultBeautifier>();
    ResultCoefficients avg_coefficients = new ResultCoefficients("average of all results");

    public void add(ResultBeautifier result) {
        this.results.add(result);
    }

    public String printAllResults() {
        StringBuilder sb = new StringBuilder();
        for (ResultBeautifier result : results) {
            sb.append(result.getResult());
            avg_coefficients.countAnother(result.coefficients);
            avg_coefficients.addResults(result.coefficients);
        }
        avg_coefficients.devideResults(results.size());
        avg_coefficients.generateCountResults(sb);
        avg_coefficients.generateResults(sb);

        return sb.toString();
    }




}
