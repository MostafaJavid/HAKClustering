import weka.clusterers.HierarchicalClusterer;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Mostafa on 2/1/2016.
 */
public class CustomClusters {
    private Instances data;
    private HierarchicalClusterer hierarchicalClusterer;
    private int minimumInstanceCount;
    private List<CustomCluster> customClusterList = new ArrayList<CustomCluster>();
    private Instances instances;
    private DistanceFunction m_DistanceFunction = new EuclideanDistance();

    public CustomClusters(Instances data, HierarchicalClusterer hierarchicalClusterer, int minimumInstanceCount) {
        this.hierarchicalClusterer = hierarchicalClusterer;
        this.data = data;
        this.minimumInstanceCount = minimumInstanceCount;
        this.m_DistanceFunction = hierarchicalClusterer.getDistanceFunction();
        getCustomClusterList();
    }

    private List<CustomCluster> computeClusters() {

        for (int i = 0; i < hierarchicalClusterer.nClusterID.length; i++) {
            if (hierarchicalClusterer.nClusterID[i].size() >= minimumInstanceCount) {
                CustomCluster cluster = new CustomCluster(i);
                for (Integer index : hierarchicalClusterer.nClusterID[i]) {
                    cluster.addInstance(data.instance(index));
                }
                customClusterList.add(cluster);
            }
        }
        return customClusterList;
    }

    public  double computeIntraClusterVariance(){
        List<CustomCluster> customClusters = getCustomClusterList();
        double result = 0;
        for (CustomCluster customCluster : customClusters) {
            result += customCluster.computeSigma(m_DistanceFunction);
        }
        return result / customClusters.size();
    }

    public double computeInterClusterVariance(){
        double result = 0;
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            result += customCluster.computeInterClusterVariance(m_DistanceFunction, centroids);
        }
        return result / getCustomClusterList().size();
    }

    public double computeFisher(){
        double result = 0;
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma= customCluster.computeSigma(m_DistanceFunction);
            double variance = customCluster.computeInterClusterVariance(m_DistanceFunction,centroids);
            result += (variance / sigma);
        }
        return result / getCustomClusterList().size();
    }

    public double computeTotalMinimumDistance(){
        double result = 0;
        CustomCluster globalCluster = new CustomCluster(-1);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma= customCluster.computeSigma(m_DistanceFunction);
            globalCluster.addInstance(customCluster.getCentroid());
            result += sigma;
        }
        result += globalCluster.computeSigma(m_DistanceFunction);
        return result;
    }


    private List<Instance> getCentroids() {
        List<Instance> centroids = new ArrayList<Instance>();
        for (CustomCluster customCluster : getCustomClusterList()) {
            centroids.add(customCluster.getCentroid());
        }
        return centroids;
    }

    private Instances convertCentroidsToInstances(Instances data, List<CustomCluster> customClusters) {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < data.numAttributes(); i++) {
            attributes.add(data.attribute(i));
        }
        Instances centroids = new Instances("centroids", attributes, customClusters.size());
        for (CustomCluster cluster : customClusters) {
            centroids.add(cluster.getCentroid());
        }
        return centroids;
    }

    private List<CustomCluster> getCustomClusterList(){
        if (customClusterList == null)
            customClusterList = computeClusters();
        return customClusterList;
    }
    public CustomCluster[] getCustomClusterArray() {
        return getCustomClusterList().toArray(new CustomCluster[0]);
    }

    public Instances getCentroidsAsInstances(){
        if (instances == null)
            instances = convertCentroidsToInstances(data,getCustomClusterList());
        return instances;
    }

    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }

}
