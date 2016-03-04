import weka.clusterers.ClusterEvaluation;
import weka.core.DistanceFunction;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Mostafa on 1/26/2016.
 */
public class CustomCluster {
    private int clusterId;
    private List<Instance> instanceList = new ArrayList<Instance>();
    private Instance centroid;
    private double withinClusterDistance =-1;
    private SupervisedEvaluater supervisedEvaluater;

    public CustomCluster(int id,ClusterEvaluation eval){
        this.clusterId = id;
        if (eval != null){
            this.supervisedEvaluater = new SupervisedEvaluater(eval,id);
        }
    }
//    public CustomCluster(int id,Instance[] instances){
//        this(id);
//        for (Instance instance : instances) {
//            addInstance(instance);
//        }
//    }

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

    public double computeDistanceFromCentroid(DistanceFunction m_DistanceFunction) {
        if (withinClusterDistance == -1 ) {
            withinClusterDistance = 0;
            Instance centroid = getCentroid();
            for (Instance instance : instanceList) {
                withinClusterDistance += m_DistanceFunction.distance(centroid, instance);
            }
            withinClusterDistance /= getInstanceList().length;
        }
        return withinClusterDistance;
    }

    private void resetComputes() {
        this.centroid = null;
        this.withinClusterDistance = -1;
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

    public SupervisedEvaluater getSupervisedEvaluater() {
        return supervisedEvaluater;
    }
}
