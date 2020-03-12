package ai.djl.jsr381.spi;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import javax.imageio.ImageIO;
import javax.visrec.ImageFactory;
import javax.visrec.spi.ImageFactoryService;

/**
 * DJL implementation of {@link ImageFactoryService} which serves the implementations of {@link
 * ImageFactory}.
 *
 * @author Frank Liu
 */
public final class DjlImageFactoryService implements ImageFactoryService {

    private static final Map<Class<?>, ImageFactory<?>> IMAGE_FACTORIES = new ConcurrentHashMap<>();

    static {
        IMAGE_FACTORIES.put(BufferedImage.class, new ImageFactoryImpl());
    }

    /**
     * Get the {@link ImageFactory} by image type.
     *
     * @param imageCls image type in {@link Class} object which is able to be processed by the image
     *     factory implementation.
     * @param <T> image type.
     * @return {@link ImageFactory} wrapped in {@link Optional}. If the {@link ImageFactory} could
     *     not be found then the {@link Optional} would contain null.
     */
    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<ImageFactory<T>> getByImageType(Class<T> imageCls) {
        Objects.requireNonNull(imageCls, "imageCls == null");
        ImageFactory<?> imageFactory = IMAGE_FACTORIES.get(imageCls);
        return Optional.ofNullable((ImageFactory<T>) imageFactory);
    }

    /** {@link ImageFactory} to provide {@link BufferedImage} as return object. */
    public static final class ImageFactoryImpl implements ImageFactory<BufferedImage> {

        /** {@inheritDoc} */
        @Override
        public BufferedImage getImage(File file) throws IOException {
            BufferedImage img = ImageIO.read(file);
            if (img == null) {
                throw new IOException(
                        "Unable to transform File into BufferedImage due to unknown image encoding");
            }
            return img;
        }

        /** {@inheritDoc} */
        @Override
        public BufferedImage getImage(URL file) throws IOException {
            BufferedImage img = ImageIO.read(file);
            if (img == null) {
                throw new IOException(
                        "Unable to transform URL into BufferedImage due to unknown image encoding");
            }
            return img;
        }

        /** {@inheritDoc} */
        @Override
        public BufferedImage getImage(InputStream file) throws IOException {
            BufferedImage img = ImageIO.read(file);
            if (img == null) {
                throw new IOException(
                        "Unable to transform InputStream into BufferedImage due to unknown image encoding");
            }
            return img;
        }
    }
}
