import java.util.Map;
import java.util.Set;

/**
 * Created by Mostafa on 2/23/2016.
 */
public class ResultCoefficients {
    private String title;
    private  ResultCoefficient cohision = new ResultCoefficient("Cohision(Smaller)",true);
    private  ResultCoefficient daviesBouldin = new ResultCoefficient("daviesBouldin(smaller)",true);
    private  ResultCoefficient dunn = new ResultCoefficient("dunn(greater)",false);
    private  ResultCoefficient silhouette = new ResultCoefficient("silhouette(greater)",false);

    public ResultCoefficients(String title){
       this.title = title;
    }

    public void countAnother(ResultCoefficients another){
        this.cohision.countAnother(another.cohision);
        this.daviesBouldin.countAnother(another.daviesBouldin);
        this.dunn.countAnother(another.dunn);
        this.silhouette.countAnother(another.silhouette);
    }


    public void generateCountResults(StringBuilder sb){
        sb.append("-------").append("Count result").append("--------").append("\n");
        cohision.generateCountResult(sb);
        daviesBouldin.generateCountResult(sb);
        dunn.generateCountResult(sb);
        silhouette.generateCountResult(sb);
        sb.append("\n\n");
        sb.toString();
    }

    public void generateResults(StringBuilder sb){
        sb.append("-------").append(title).append("--------").append("\n");
        cohision.generateResult(sb);
        daviesBouldin.generateResult(sb);
        dunn.generateResult(sb);
        silhouette.generateResult(sb);
        sb.append("\n\n");
        sb.toString();
    }

    public void addResults(ResultCoefficients another){
        addResults(this.getCohision(), another.getCohision().entrySet());
        addResults(this.getDaviesBouldin(), another.getDaviesBouldin().entrySet());
        addResults(this.getDunn(), another.getDunn().entrySet());
        addResults(this.getSilhouette(), another.getSilhouette().entrySet());
    }

    private void addResults(ResultCoefficient coefficient, Set<Map.Entry<String,Double>> entrySet) {
        for (Map.Entry<String, Double> entry : entrySet) {
            if (!coefficient.containsKey(entry.getKey()))
                coefficient.put(entry.getKey(), (double) 0);
            double value = coefficient.get(entry.getKey()) + entry.getValue();
            coefficient.put(entry.getKey(), value);
        }
    }

    public void devideResults(Integer size){
        devideResults(size,cohision);
        devideResults(size,daviesBouldin);
        devideResults(size,dunn);
        devideResults(size,silhouette);
    }

    private void devideResults(Integer size,ResultCoefficient coefficient){
        for (Map.Entry<String, Double> entry : coefficient.entrySet()) {
            coefficient.put(entry.getKey(),entry.getValue() / size);
        }
    }

    public ResultCoefficient getCohision() {
        return cohision;
    }

    public ResultCoefficient getDaviesBouldin() {
        return daviesBouldin;
    }

    public ResultCoefficient getDunn() {
        return dunn;
    }

    public ResultCoefficient getSilhouette() {
        return silhouette;
    }
}
