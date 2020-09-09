package ai.djl.jsr381.dataset;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class CsvDataset extends RandomAccessDataset {

    private List<CSVRecord> records;

    private CsvDataset(Builder builder) {
        super(builder);
        records = builder.records;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Record get(NDManager manager, long index) {
        CSVRecord record = records.get(Math.toIntExact(index));
        int size = record.size();
        float[] data = new float[size - 1];
        for (int i = 0; i < size - 1; ++i) {
            data[i] = Float.parseFloat(record.get(i));
        }
        NDArray datum = manager.create(data);
        NDArray label = manager.create(Float.parseFloat(record.get(size - 1)));
        return new Record(new NDList(datum), new NDList(label));
    }

    @Override
    public long availableSize() {
        return records.size();
    }

    @Override
    public void prepare(Progress progress) {}

    public static final class Builder extends BaseBuilder<Builder> {

        List<CSVRecord> records;

        private File file;

        @Override
        protected Builder self() {
            return this;
        }

        public Builder setCsvFile(File file) {
            this.file = file;
            return this;
        }

        public CsvDataset build() throws IOException {
            try (Reader reader = Files.newBufferedReader(file.toPath());
                    CSVParser csvParser =
                            new CSVParser(
                                    reader,
                                    CSVFormat.DEFAULT
                                            .withFirstRecordAsHeader()
                                            .withIgnoreHeaderCase()
                                            .withTrim())) {
                records = csvParser.getRecords();
            }
            return new CsvDataset(this);
        }
    }
}
