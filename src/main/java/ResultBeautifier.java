import weka.clusterers.ClusterEvaluation;
import weka.core.Utils;

import java.util.*;

/**
 * Created by Mostafa on 2/14/2016.
 */
public class ResultBeautifier {
    String title;
    ResultCoefficients coefficients = new ResultCoefficients("current comparison");

    List<CustomClusters> results;

    public ResultBeautifier(String title, List<CustomClusters> results) {
        this.title = title;
        this.results = results;
    }

    public String getResult() {
        StringBuilder sb = new StringBuilder();
        sb.append("------------------------------------------------------------------------").append("\n");
        sb.append("------------------------------").append(title).append("------------------------------------------").append("\n");
        sb.append("------------------------------------------------------------------------").append("\n");
        for (CustomClusters result : results) {
            sb.append(getResultString(result));
        }
        coefficients.generateResults(sb);
        return sb.toString();
    }

    /////////Build Result String//////////////////////////////////////////////
    private String getResultString(CustomClusters customClusters) {
        StringBuilder sb = new StringBuilder();
        String methodName = customClusters.getTitle(); //customClusters.getClusterer().getClass().getSimpleName();
        sb.append("**************************").append(methodName).append("************************").append("\n");
        if (customClusters.getClusterer() instanceof HAKClusterer) {
            sb.append(getHakCustomizedResult((HAKClusterer) customClusters.getClusterer()));
        }
        sb.append("centroids count:").append(customClusters.getCustomClusterList().size()).append("\n");
        for (CustomCluster customCluster : customClusters.getCustomClusterList()) {
            sb.append("size of cluster ").append(customCluster.getClusterId()).append(":").append(customCluster.getClusterCount()).append("\n");
        }
        sb.append(getSupervisedResult(customClusters.getEval(), customClusters));

        coefficients.printCoefficientsResults(customClusters, sb, methodName);
        sb.append("\n");
        return sb.toString();
    }

    private static String getHakCustomizedResult(HAKClusterer hakClusterer) {
        StringBuilder sb = new StringBuilder();

        sb.append("link type: ").append(hakClusterer.getLinkType().getSelectedTag().getReadable()).append("\n");
        sb.append("max iteration: ").append(hakClusterer.getMaxIteration()).append("\n");
        sb.append("outlier factor: ").append(hakClusterer.getOutlierFactor()).append("\n");
        sb.append("outlier min dense count: ").append(hakClusterer.getOutlierMinDense()).append("\n");
        sb.append("outliers count: ").append(hakClusterer.getOutliers().keySet().size()).append("\n");
//        for (Map.Entry<Integer, Integer> outlier : hakClusterer.getOutliers().entrySet()) {
//            sb.append(outlier.getKey()).append(":").append(outlier.getValue()).append("\n");
//        }
        getMiddleCentroidString(sb, "middle HAK Centroids-1:", hakClusterer.getMiddleClusters());
        getMiddleCentroidString(sb, "middle HAK Centroids-2:", hakClusterer.getMiddleClusters2());
        return sb.toString();
    }

    private static void getMiddleCentroidString(StringBuilder sb, String title, CustomClusters middleClusters) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        sb.append(title).append(middleClusters.getCustomClusterList().size()).append("\n");
        for (CustomCluster customCluster : middleClusters.getCustomClusterList()) {
            int count = customCluster.getClusterCount();
            if (map.containsKey(count)) {
                map.put(count, map.get(count) + 1);
            } else {
                map.put(count, 1);
            }
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            sb.append("number of clusters with size of ").append(entry.getKey()).append(":").append(entry.getValue()).append("\n");
        }
    }

    private String getSupervisedResult(ClusterEvaluation eval, CustomClusters customClusters) {
        StringBuilder sb = new StringBuilder();
        sb.append("Incorrectly clustered instances :\t"
                + eval.best[eval.getNumClusters()]
                + "\t"
                + (Utils.doubleToString((eval.best[eval.getNumClusters()] / eval.numInstances * 100.0), 8,
                4)) + " %\n");
        //sb.append(eval.clusterResultsToString());
        for (CustomCluster customCluster : customClusters.getCustomClusterList()) {
            sb.append(customCluster.getSupervisedEvaluater()).append("\n");
        }
        return sb.toString();
    }

    public String getTitle() {
        return title;
    }
}
