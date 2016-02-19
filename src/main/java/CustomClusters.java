import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

import java.util.*;

/**
 * Created by Mostafa on 2/1/2016.
 */
public class CustomClusters {
    private String title;
    private Instances data;
    private AbstractClusterer clusterer;
    private int minimumInstanceCount;
    private List<CustomCluster> customClusterList;
    private Instances centroids;
    private DistanceFunction m_DistanceFunction = new EuclideanDistance();
    private Map<Integer,Integer> dataToClustersAssignments = new HashMap<Integer, Integer>();
    private Map<Integer,List<Integer>> clusterToDataAssignments = new HashMap<Integer, List<Integer>>();
    private Map<Integer,Double> silhouetteMap_data = new HashMap<Integer, Double>();
    private Map<Integer,Double> silhouetteMap_cluster = new HashMap<Integer, Double>();
    double[][] betweenAllDataDistances;
    double[][] betweenClusterCentroidDistances;
    ClusterEvaluation eval;

    //////Props////////////////////////////////////////////////////////////
    public List<CustomCluster> getCustomClusterList() {
        return customClusterList;
    }

    public  CustomCluster[] getCustomClusterArray() {
        return getCustomClusterList().toArray(new CustomCluster[0]);
    }

    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }

    public AbstractClusterer getClusterer() {
        return clusterer;
    }

    public ClusterEvaluation getEval() {
        return eval;
    }

    //////Constructor////////////////////////////////
    public CustomClusters(String title,Instances data, AbstractClusterer clusterer, int minimumInstanceCount,ClusterEvaluation eval) {
        this.title = title;
        this.data = data;
        this.minimumInstanceCount = minimumInstanceCount;
        String[] options = new String[2];
        options[0] = "-D";
        options[1] = "true";
        this.m_DistanceFunction = new EuclideanDistance(); //hierarchicalClusterer.getDistanceFunction();
        try {
            this.m_DistanceFunction.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.clusterer = clusterer;
        if (clusterer instanceof HierarchicalClusterer) {
            HierarchicalClusterer hierarchicalClusterer = (HierarchicalClusterer) clusterer;
            this.m_DistanceFunction.setInstances(hierarchicalClusterer.getDistanceFunction().getInstances());
            this.customClusterList = computeClusters(hierarchicalClusterer);
        } else {
            SimpleKMeans simpleKMeans = (SimpleKMeans) clusterer;
            this.m_DistanceFunction.setInstances(simpleKMeans.getDistanceFunction().getInstances());
            this.customClusterList = computeClusters(simpleKMeans);
        }
        initializeBetweenClusterDistances();
        this.eval = eval;
    }

    ////////Methods/////////////////////////////////////////
    private List<Instance> getCentroids() {
        List<Instance> centroids = new ArrayList<Instance>();
        for (CustomCluster customCluster : getCustomClusterList()) {
            centroids.add(customCluster.getCentroid());
        }
        return centroids;
    }

    public static Instances convertToInstances(List<CustomCluster> list, Instances data) {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < data.numAttributes(); i++) {
            attributes.add(data.attribute(i));
        }
        Instances centroids = new Instances("centroids", attributes, list.size());
        for (CustomCluster cluster : list) {
            centroids.add(cluster.getCentroid());
        }
        return centroids;
    }

    private List<CustomCluster> sort(List<CustomCluster> list){
        Collections.sort(list, new Comparator<CustomCluster>() {
            @Override
            public int compare(CustomCluster o1, CustomCluster o2) {
                return o1.getClusterCount() > o2.getClusterCount() ? -1 : 1;
            }
        });
        return list;
    }

    public void filterClusters(int maxClustersCount) {
        customClusterList = sort(getCustomClusterList());
        List<CustomCluster> result = new ArrayList<CustomCluster>();
        for (int i = 0; i < maxClustersCount; i++) {
            result.add(customClusterList.get(i));
        }
        customClusterList = result;
    }

    public Instances getCentroidsAsInstances() {
        if (centroids == null) {
            centroids = convertToInstances(getCustomClusterList(),data);
        }
        return centroids;
    }

    private List<CustomCluster> computeClusters(HierarchicalClusterer hierarchicalClusterer) {

        List<CustomCluster> result = new ArrayList<CustomCluster>();
        int clusterIndex=-1;
        for (int i = 0; i < hierarchicalClusterer.nClusterID.length; i++) {
            if (hierarchicalClusterer.nClusterID[i].size() >= minimumInstanceCount) {
                clusterIndex++;
                CustomCluster cluster = new CustomCluster(i);
                for (Integer index : hierarchicalClusterer.nClusterID[i]) {
                    cluster.addInstance(data.instance(index));
                    dataToClustersAssignments.put(index, clusterIndex);
                }
                result.add(cluster);
            }
        }
        return result;
    }

    private List<CustomCluster> computeClusters(SimpleKMeans kMeans) {
        try {
            int[] temp = kMeans.getAssignments();
            for (int i = 0; i < temp.length; i++) {
                dataToClustersAssignments.put(i, temp[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        List<CustomCluster> result = new ArrayList<CustomCluster>();
        for (int i = 0; i < kMeans.tempI.length; i++) {
            if (kMeans.tempI[i].size() >= minimumInstanceCount) {
                CustomCluster cluster = new CustomCluster(i);
                for (Instance instance : kMeans.tempI[i]) {
                    cluster.addInstance(instance);
                }
                result.add(cluster);
            }
        }
        return result;
    }

    //////////Evaluation/////////////////////////////////////////////////
    public double computeWithinClusterVariance() {
        List<CustomCluster> customClusters = getCustomClusterList();
        double result = 0;
        for (CustomCluster customCluster : customClusters) {
            result += customCluster.computeDistanceFromCentroid(m_DistanceFunction);
        }
        result /= customClusters.size();
        return result;
    }

    public double computeBetweenClusterVariance() {
        double result = 0;
        for (int i = 0; i < customClusterList.size(); i++) {
            result += computeCentroidDistanceFromOtherCentroids(i);
        }
        result /= getCustomClusterList().size();
        return result;
    }


    private double computeCentroidDistanceFromOtherCentroids(int currentClusterIdx){
        double result = 0;
        for (int i = 0; i < customClusterList.size(); i++) {
            result += betweenClusterCentroidDistances[currentClusterIdx][i];
        }
        result /= customClusterList.size();
        return result;
    }

    public double computeFisher() {
        double result = 0;
        for (int i = 0; i < customClusterList.size(); i++) {
            double sigma = customClusterList.get(i).computeDistanceFromCentroid(m_DistanceFunction);
            double variance = computeCentroidDistanceFromOtherCentroids(i);
            result += (variance / sigma);
        }
        result /= getCustomClusterList().size();
        return result;
    }

    public double computeTotalMinimumDistance() {
        double result = 0;
        CustomCluster globalCluster = new CustomCluster(-1);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma = customCluster.computeDistanceFromCentroid(m_DistanceFunction);
            globalCluster.addInstance(customCluster.getCentroid());
            result += sigma;
        }
        result += globalCluster.computeDistanceFromCentroid(m_DistanceFunction);
        return result;
    }

    // the smallest Davies–Bouldin index is considered the best algorithm
    public double computeDaviesBouldin(){
        double result = 0;
        for (int i = 0; i < customClusterList.size(); i++) {
            double maxDiff = Double.MIN_VALUE;
            for (int j = 0; j < customClusterList.size(); j++) {
                if (i != j){
                    double sigma_i = customClusterList.get(i).computeDistanceFromCentroid(m_DistanceFunction);
                    double sigma_j = customClusterList.get(j).computeDistanceFromCentroid(m_DistanceFunction);
                    double d_i_j = betweenClusterCentroidDistances[i][j]; //m_DistanceFunction.distance(customClusterList.get(i).getCentroid(),customClusterList.get(j).getCentroid());
                    double diff = (sigma_i + sigma_j) / d_i_j;
                    if (diff > maxDiff){
                        maxDiff = diff;
                    }
                }
            }
            result += maxDiff;
        }
        result /= customClusterList.size();
        return result;
    }

    // algorithms that produce clusters with high Dunn index are more desirable.
    public double computeDunn(){
        double result =0;
        double minBetweenClusterCentroidDistance = Double.MAX_VALUE;
        for (int i = 0; i < customClusterList.size(); i++) {
            for (int j = 0; j < customClusterList.size(); j++) {
                if (i!=j && betweenClusterCentroidDistances[i][j] < minBetweenClusterCentroidDistance){
                    minBetweenClusterCentroidDistance = betweenClusterCentroidDistances[i][j];
                }
            }
        }
        double maxWithinCluster = Double.MIN_VALUE;
        for (int i = 0; i < customClusterList.size(); i++) {
            double temp = customClusterList.get(i).computeDistanceFromCentroid(m_DistanceFunction);
            if (temp > maxWithinCluster){
                maxWithinCluster = temp;
            }
        }
        result = minBetweenClusterCentroidDistance / maxWithinCluster;
        return result;
    }

    //The silhouette coefficient contrasts the average distance to elements in the same cluster with the average distance to elements in other clusters.
    //Objects with a high silhouette value are considered well clustered, objects with a low value may be outliers.
    //This index works well with k-means clustering, and is also used to determine the optimal number of clusters.
    public double computeSilhouette(){
        initializeBetweenAllDataDistances();
        initializeClusterToDataAssignments();
        silhouetteMap_data = new HashMap<Integer, Double>();
        silhouetteMap_cluster = new HashMap<Integer, Double>();
        for (int i = 0; i < data.size(); i++) {
            if (dataToClustersAssignments.containsKey(i)){
                int clusterNo = dataToClustersAssignments.get(i);
                double a_i = get_a_i(i, clusterNo);
                double b_i = get_b_i(i, clusterNo);
                double s_i = (b_i - a_i) / (Math.max(a_i,b_i));
                silhouetteMap_data.put(i,s_i);
            }
        }
        for (Integer clusterNo : clusterToDataAssignments.keySet()) {
            double clusterSilhouette = 0;
            for (Integer  instanceIdx : clusterToDataAssignments.get(clusterNo)) {
                clusterSilhouette += silhouetteMap_data.get(instanceIdx);
            }
            clusterSilhouette /= clusterToDataAssignments.get(clusterNo).size();
            silhouetteMap_cluster.put(clusterNo,clusterSilhouette);
        }
        double result = 0;
        for (Double dataSilhouette : silhouetteMap_data.values()) {
            result += dataSilhouette;
        }
        result /= (silhouetteMap_data.values().size());
//        double result = 0;
//        for (Double dataSilhouette : silhouetteMap_cluster.values()) {
//            result += dataSilhouette;
//        }
//        result /= (silhouetteMap_cluster.values().size());
        return result;
    }

    private double get_b_i(int i, int clusterNo) {
        double b_i =Double.MAX_VALUE;
        for (Integer otherClusterNo : clusterToDataAssignments.keySet()) {
            if (clusterNo != otherClusterNo){
                double temp = 0;
                for (Integer otherClusterInstanceIdx : clusterToDataAssignments.get(otherClusterNo)) {
                    temp += betweenAllDataDistances[i][otherClusterInstanceIdx];
                }
                temp /= clusterToDataAssignments.get(otherClusterNo).size();
                if (temp < b_i){
                    b_i = temp;
                }
            }
        }
        return b_i;
    }

    private double get_a_i(int i, int clusterNo) {
        double a_i = 0;
        for (Integer instanceIdx : clusterToDataAssignments.get(clusterNo)) {
            a_i += betweenAllDataDistances[i][instanceIdx];
        }
        a_i /= (clusterToDataAssignments.get(clusterNo).size()-1);
        return a_i;
    }

    private void initializeClusterToDataAssignments() {
        for (Map.Entry<Integer, Integer> entry : dataToClustersAssignments.entrySet()) {
            if (!clusterToDataAssignments.containsKey(entry.getValue())){
                clusterToDataAssignments.put(entry.getValue(),new ArrayList<Integer>());
            }
            clusterToDataAssignments.get(entry.getValue()).add(entry.getKey());
        }
    }

    private void initializeBetweenAllDataDistances() {
        this.betweenAllDataDistances = new double[data.size()][data.size()];
        for (Integer id : dataToClustersAssignments.keySet()) {
            for (Integer other : dataToClustersAssignments.keySet()) {
                if (id != other && betweenAllDataDistances[id][other] == 0){
                    betweenAllDataDistances[id][other] = getDistanceFunction().distance(data.instance(id),data.instance(other));
                    betweenAllDataDistances[other][id] = betweenAllDataDistances[id][other];
                }
            }
        }
    }

    private void initializeBetweenClusterDistances(){
        betweenClusterCentroidDistances = new double[customClusterList.size()][customClusterList.size()];
        for (int i = 0; i < customClusterList.size(); i++) {
            for (int j = 0; j < customClusterList.size(); j++) {
                if (i!=j){
                    if (betweenClusterCentroidDistances[i][j] == 0){
                        betweenClusterCentroidDistances[i][j] = m_DistanceFunction.distance(customClusterList.get(i).getCentroid(),customClusterList.get(j).getCentroid());
                        betweenClusterCentroidDistances[j][i] = betweenClusterCentroidDistances[i][j];
                    }
                }
            }
        }
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }
}
