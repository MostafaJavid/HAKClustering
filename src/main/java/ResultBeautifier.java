import weka.clusterers.ClusterEvaluation;
import weka.core.Utils;

import java.util.*;

/**
 * Created by Mostafa on 2/14/2016.
 */
public class ResultBeautifier {
    Map<String, Double> cohisionMap = new HashMap<String, Double>();
    Map<String, Double> daviesBouldinMap = new HashMap<String, Double>();
    Map<String, Double> dunnMap = new HashMap<String, Double>();
    Map<String, Double> silhoetteMap = new HashMap<String, Double>();

    List<CustomClusters> results;

    public ResultBeautifier(List<CustomClusters> results) {
        this.results = results;
    }

    public String getResult() {
        StringBuilder sb = new StringBuilder();
        for (CustomClusters result : results) {
            sb.append(getResultString(result));
        }
        sb.append("-------Comparison--------").append("\n");
        getComparisonCoefficent(sb, "Cohision(smaller)", true, cohisionMap);
        getComparisonCoefficent(sb, "Davies-Bouldin(smaller)", true, daviesBouldinMap);
        getComparisonCoefficent(sb,"Dunn(greater)",false,dunnMap);
        getComparisonCoefficent(sb,"Silhoette(greater)",false,silhoetteMap);
        return sb.toString();
    }

    private void getComparisonCoefficent(StringBuilder sb,String title, final boolean isAscending,Map<String,Double> map) {
        List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                int factor = isAscending ? 1 : -1;
                return o1.getValue() > o2.getValue() ? 1 * factor : (o1.getValue() < o2.getValue() ? -1 * factor : 0);
            }
        });
        sb.append(title).append(":");
        for (Map.Entry<String, Double> entry : list) {
            sb.append(entry.getKey()).append("(").append(entry.getValue()).append(") - ");
        }
        sb.append("\n");
    }

    /////////Build Result String//////////////////////////////////////////////
    private String getResultString(CustomClusters customClusters) {
        StringBuilder sb = new StringBuilder();
        String methodName = customClusters.getTitle(); //customClusters.getClusterer().getClass().getSimpleName();
        sb.append("******************************************************************************").append("\n");
        sb.append(methodName).append("\n");
        if (customClusters.getClusterer() instanceof HAKClusterer) {
            sb.append(getHakCustomizedResult((HAKClusterer) customClusters.getClusterer()));
        }
        sb.append("centroids count:").append(customClusters.getCustomClusterList().size()).append("\n");
        for (CustomCluster customCluster : customClusters.getCustomClusterList()) {
            sb.append("size of cluster ").append(customCluster.getClusterId()).append(":").append(customCluster.getClusterCount()).append("\n");
        }
        sb.append(getSupervisedResult(customClusters.getEval()));
        cohisionMap.put(methodName,customClusters.computeWithinClusterVariance());
        sb.append("Average of within cluster variance():").append(cohisionMap.get(methodName)).append("\n");
//        sb.append("between cluster variance:").append(computeBetweenClusterVariance()).append("\n");
//        sb.append("Fisher:").append(computeFisher()).append("\n");
//        sb.append("total distance:").append(computeTotalMinimumDistance()).append("\n");
        daviesBouldinMap.put(methodName, customClusters.computeDaviesBouldin());
        sb.append("Davies–Bouldin(smaller):").append(daviesBouldinMap.get(methodName)).append("\n");
        dunnMap.put(methodName, customClusters.computeDunn());
        sb.append("Dunn(greater):").append(dunnMap.get(methodName)).append("\n");
        silhoetteMap.put(methodName, customClusters.computeSilhouette());
        sb.append("silhouette(greater):").append(silhoetteMap.get(methodName)).append("\n");
        sb.append("\n");
        return sb.toString();
    }

    private static String getHakCustomizedResult(HAKClusterer hakClusterer) {
        StringBuilder sb = new StringBuilder();

        sb.append("link type: ").append(hakClusterer.getLinkType().getSelectedTag().getReadable()).append("\n");
        sb.append("max iteration: ").append(hakClusterer.getMaxIteration()).append("\n");
        sb.append("outlier factor: ").append(hakClusterer.getOutlierFactor()).append("\n");
        sb.append("outlier min dense count: ").append(hakClusterer.getOutlierMinDense()).append("\n");
        sb.append("outliers: ").append("\n");
        for (Map.Entry<Integer, Integer> outlier : hakClusterer.getOutliers().entrySet()) {
            sb.append(outlier.getKey()).append(":").append(outlier.getValue()).append("\n");
        }
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

    public String getSupervisedResult(ClusterEvaluation eval) {
        StringBuilder sb = new StringBuilder();
        sb.append("Incorrectly clustered instances :\t"
                + eval.best[eval.getNumClusters()]
                + "\t"
                + (Utils.doubleToString((eval.best[eval.getNumClusters()] / eval.numInstances * 100.0), 8,
                4)) + " %\n");
        return sb.toString();
    }

}
