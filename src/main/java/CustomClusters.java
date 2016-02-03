import weka.clusterers.AbstractClusterer;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Mostafa on 2/1/2016.
 */
public class CustomClusters {
    private Instances data;
    private AbstractClusterer clusterer;
    private int minimumInstanceCount;
    private List<CustomCluster> customClusterList;
    private Instances instances;
    private DistanceFunction m_DistanceFunction = new EuclideanDistance();

    public CustomClusters(Instances data, AbstractClusterer clusterer, int minimumInstanceCount) {
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
    }

    private List<CustomCluster> computeClusters(HierarchicalClusterer hierarchicalClusterer) {

        List<CustomCluster> result = new ArrayList<CustomCluster>();
        for (int i = 0; i < hierarchicalClusterer.nClusterID.length; i++) {
            if (hierarchicalClusterer.nClusterID[i].size() >= minimumInstanceCount) {
                CustomCluster cluster = new CustomCluster(i);
                for (Integer index : hierarchicalClusterer.nClusterID[i]) {
                    cluster.addInstance(data.instance(index));
                }
                result.add(cluster);
            }
        }
        return result;
    }

    private List<CustomCluster> computeClusters(SimpleKMeans kMeans) {
        String[] options = new String[2];
        options[0] = "-D";
        options[1] = "true";
        this.m_DistanceFunction = new EuclideanDistance(); //hierarchicalClusterer.getDistanceFunction();
        try {
            this.m_DistanceFunction.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.m_DistanceFunction.setInstances(kMeans.getDistanceFunction().getInstances());

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

    public double computeIntraClusterVariance() {
        List<CustomCluster> customClusters = getCustomClusterList();
        double result = 0;
        for (CustomCluster customCluster : customClusters) {
            result += customCluster.computeSigma(m_DistanceFunction);
        }
        result /= customClusters.size();
        return result;
    }

    public double computeInterClusterVariance() {
        double result = 0;
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            result += customCluster.computeInterClusterVariance(m_DistanceFunction, centroids);
        }
        result /= getCustomClusterList().size();
        return result;
    }

    public double computeFisher() {
        double result = 0;
        Instance[] centroids = getCentroids().toArray(new Instance[0]);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma = customCluster.computeSigma(m_DistanceFunction);
            double variance = customCluster.computeInterClusterVariance(m_DistanceFunction, centroids);
            result += (variance / sigma);
        }
        result /= getCustomClusterList().size();
        return result;
    }

    public double computeTotalMinimumDistance() {
        double result = 0;
        CustomCluster globalCluster = new CustomCluster(-1);
        for (CustomCluster customCluster : getCustomClusterList()) {
            double sigma = customCluster.computeSigma(m_DistanceFunction);
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

    private List<CustomCluster> getCustomClusterList() {
        return customClusterList;
    }

    public CustomCluster[] getCustomClusterArray() {
        return getCustomClusterList().toArray(new CustomCluster[0]);
    }

    public Instances getCentroidsAsInstances() {
        if (instances == null)
            instances = convertCentroidsToInstances(data, getCustomClusterList());
        return instances;
    }

    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }


    public String getResultString() {
        StringBuilder sb = new StringBuilder();
        sb.append("******************************************************************************").append("\n");
        sb.append("intra cluster variance:").append(computeIntraClusterVariance()).append("\n");
        sb.append("inter cluster variance:").append(computeInterClusterVariance()).append("\n");
        sb.append("Fisher:").append(computeFisher()).append("\n");
        sb.append("total distance:").append(computeTotalMinimumDistance()).append("\n");
        return sb.toString();
    }

}
