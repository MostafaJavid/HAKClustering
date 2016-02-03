import weka.core.DistanceFunction;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by Mostafa on 1/26/2016.
 */
public class CustomCluster {
    private int clusterId;
    private List<Instance> instanceList = new ArrayList<Instance>();
    private Instance centroid;
    private double sigma=-1;
    private double interClusterVariance=-1;

    public CustomCluster(int id){
        this.clusterId = id;
    }
    public CustomCluster(int id,Instance[] instances){
        this(id);
        for (Instance instance : instances) {
            addInstance(instance);
        }
    }

    /////////////////////////Methods//////////////////////////////////////////////////////////////
    private Instance computeCentroid() {
        int numAtts = instanceList.get(0).numAttributes();

        double[] fValues1 = new double[numAtts];
        for (int i = 0; i < instanceList.size(); i++) {
            Instance instance = instanceList.get(i);
            for (int j = 0; j < numAtts; j++) {
                fValues1[j] += instance.value(j);
            }
        }
        for (int j = 0; j < numAtts; j++) {
            fValues1[j] /= instanceList.size();
        }

        Instance centroid = (Instance) instanceList.get(0).copy();
        for (int j = 0; j < numAtts; j++) {
            centroid.setValue(j, fValues1[j]);
        }
        return centroid;
    }

    public double computeSigma(DistanceFunction m_DistanceFunction) {
        if (sigma == -1 ) {
            sigma = computeDistance(m_DistanceFunction, getCentroid());
        }
        return sigma;
    }

    private double computeDistance(DistanceFunction m_DistanceFunction,Instance centroid) {
        double result = 0;
        for (Instance instance : instanceList) {
            result += Math.pow(m_DistanceFunction.distance(centroid, instance),2);
        }
         result /= getInstanceList().length;
        return result;
    }

    public double computeInterClusterVariance(DistanceFunction m_DistanceFunction, Instance[] centroids){
        if (interClusterVariance == -1 ||  Double.isNaN(interClusterVariance)) {
            double result = 0;
            Instance myCentroid = getCentroid();
            for (Instance centroid : centroids) {
                if (myCentroid != centroid) {
                    result += computeDistance(m_DistanceFunction, centroid);
                }
            }
            interClusterVariance = result / (centroids.length - 1);
        }
        return interClusterVariance;
    }

    private void resetComputes() {
        this.centroid = null;
        this.sigma = -1;
        this.interClusterVariance =-1;
    }

    /////////////////////////PROPS////////////////////////////////////////////////////////////////
    public int getClusterId() {
        return clusterId;
    }

    public int getClusterCount() {
        return getInstanceList().length;
    }

    public void addInstance(Instance instance) {
        this.instanceList.add(instance);
        resetComputes();
    }

    public Instance getCentroid() {
        if (centroid == null)
            centroid = computeCentroid();
        return centroid;
    }

    public Instance[] getInstanceList() {
        return this.instanceList.toArray(new Instance[0]);
    }
}
