import weka.clusterers.AbstractClusterer;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

import java.util.*;

/**
 * Created by Mostafa on 2/1/2016.
 */
public class CustomClusters {
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
    double[][] allDistances;

    //////Props////////////////////////////////////////////////////////////
    private List<CustomCluster> getCustomClusterList() {
        return customClusterList;
    }

    public  CustomCluster[] getCustomClusterArray() {
        return getCustomClusterList().toArray(new CustomCluster[0]);
    }

    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }

    //////Constructor////////////////////////////////
    public CustomClusters(Instances data, AbstractClusterer clusterer, int minimumInstanceCount) {
        this.data = data;
        this.allDistances = new double[data.size()][data.size()];
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
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            result += customCluster.computeCentroidDistanceFromOtherCentroids(m_DistanceFunction, centroids);
        }
        result /= getCustomClusterList().size();
        return result;
    }

    public double computeFisher() {
        double result = 0;
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma = customCluster.computeDistanceFromCentroid(m_DistanceFunction);
            double variance = customCluster.computeCentroidDistanceFromOtherCentroids(m_DistanceFunction, centroids);
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

    public double computeSilhouette(){
        initializeAllDistances();
        initializeClusterToDataAssignments();
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
        for (Double clusterSilhouette : silhouetteMap_cluster.values()) {
            result += clusterSilhouette;
        }
        result /= (silhouetteMap_cluster.values().size());
        return result;
    }

    private double get_b_i(int i, int clusterNo) {
        double b_i =Double.MAX_VALUE;
        double temp = 0;
        for (Integer otherClusterNo : clusterToDataAssignments.keySet()) {
            if (clusterNo != otherClusterNo){
                for (Integer otherClusterInstanceIdx : clusterToDataAssignments.get(otherClusterNo)) {
                    temp += allDistances[i][otherClusterInstanceIdx];
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
            a_i += allDistances[i][instanceIdx];
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

    private void initializeAllDistances() {
        for (Integer id : dataToClustersAssignments.keySet()) {
            for (Integer other : dataToClustersAssignments.keySet()) {
                if (id != other && allDistances[id][other] == 0){
                    allDistances[id][other] = getDistanceFunction().distance(data.instance(id),data.instance(other));
                    allDistances[other][id] = allDistances[id][other];
                }
            }
        }
    }

    /////////Build Result String//////////////////////////////////////////////
    public String getResultString() {
        StringBuilder sb = new StringBuilder();
        //sb.append("******************************************************************************").append("\n");
        sb.append("centroids count:").append(getCustomClusterList().size()).append("\n");
        for (CustomCluster customCluster : getCustomClusterList()) {
            sb.append("size of cluster ").append(customCluster.getClusterId()).append(":").append(customCluster.getClusterCount()).append("\n");
        }
        sb.append("intra cluster variance:").append(computeWithinClusterVariance()).append("\n");
        sb.append("inter cluster variance:").append(computeBetweenClusterVariance()).append("\n");
        sb.append("Fisher:").append(computeFisher()).append("\n");
        sb.append("total distance:").append(computeTotalMinimumDistance()).append("\n");
        sb.append("silhouette:").append(computeSilhouette()).append("\n");
        return sb.toString();
    }

}
