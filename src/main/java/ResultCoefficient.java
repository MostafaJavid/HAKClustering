import java.util.*;

/**
 * Created by Mostafa on 2/23/2016.
 */
public class ResultCoefficient {
    private final String title;
    private final Map<String,Double> map = new HashMap<String, Double>();
    Map<String,Integer> count = new HashMap<String, Integer>();
    private final boolean isAscending;
    String bestMethod;

    public ResultCoefficient(String title,boolean isAscending){
        this.title = title;
        this.isAscending = isAscending;
    }

    public void countAnother(ResultCoefficient another){
        for (Map.Entry<String, Integer> entry : another.count.entrySet()) {
            if (this.count.containsKey(entry.getKey())) {
                this.count.put(entry.getKey(), entry.getValue() + this.count.get(entry.getKey()));
            }
            else
            {
                this.count.put(entry.getKey(),entry.getValue());
            }
        }
    }

    public void generateResult(StringBuilder sb) {
        List<Map.Entry<String, Double>> list = sort();
        bestMethod = list.get(0).getKey();
        for (Map.Entry<String, Double> entry : map.entrySet()) {
            count.put(entry.getKey(),bestMethod.compareTo(entry.getKey()) == 0 ? 1 : 0);
        }
        sb.append(getTitle()).append(":");
        for (Map.Entry<String, Double> entry : list) {
            sb.append(entry.getKey()).append("(").append(entry.getValue()).append(") - ");
        }
        sb.append("\n");
    }

    private List<Map.Entry<String, Double>> sort() {
        List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                int factor = isAscending() ? 1 : -1;
                return o1.getValue() > o2.getValue() ? 1 * factor : (o1.getValue() < o2.getValue() ? -1 * factor : 0);
            }
        });
        return list;
    }

    public void generateCountResult(StringBuilder sb){
        List<Map.Entry<String, Integer>> list = sortCount();
        sb.append(getTitle()).append(":");
        for (Map.Entry<String, Integer> entry : list) {
            sb.append(entry.getKey()).append("(").append(entry.getValue()).append(") - ");
        }
        sb.append("\n");
    }

    private List<Map.Entry<String, Integer>> sortCount() {
        List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(count.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue() > o2.getValue() ? -1 : (o1.getValue() < o2.getValue() ? 1 : 0);
            }
        });
        return list;
    }

    public String getBestMethod(){
        return bestMethod;
    }

    public void put(String methodName,Double value){
        map.put(methodName,value);
    }

    public Double get(String methodName){
        return map.get(methodName);
    }

    public String getTitle() {
        return title;
    }

    public Set<Map.Entry<String,Double>> entrySet(){
        return map.entrySet();
    }

    public Map<String, Double> getMap() {
        return map;
    }

    public boolean isAscending() {
        return isAscending;
    }

    public boolean containsKey(String key){
        return map.containsKey(key);
    }
}
