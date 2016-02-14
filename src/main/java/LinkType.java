import weka.clusterers.HierarchicalClusterer;
import weka.core.SelectedTag;

/**
 * Created by Mostafa on 2/14/2016.
 */
public enum LinkType {
    SINGLE(0,"SINGLE"),COMPLETE(1,"COMPLETE"),AVERAGE(2,"AVERAGE"),MEAN(3,"MEAN"),CENTROID(4,"CENTROID")
    ,WARD(5,"WARD"),ADJCOMPLETE(6, "ADJCOMPLETE"),NEIGHBOR_JOINING(7, "NEIGHBOR_JOINING") ;

    private SelectedTag tag;
    public SelectedTag getTag(){return tag;}
    private LinkType(int code, String name){
        this.tag = new SelectedTag(name, HierarchicalClusterer.TAGS_LINK_TYPE);
    }
}
