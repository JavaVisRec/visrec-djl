package ai.djl.jsr381.spi;

import javax.visrec.spi.BuilderService;
import javax.visrec.spi.ImageFactoryService;
import javax.visrec.spi.ImplementationService;
import javax.visrec.spi.ServiceProvider;

/**
 * {@link ServiceProvider} implementation with DJL.
 *
 * @author Frank Liu
 */
public final class DjlServiceProvider extends ServiceProvider {

    /** {@inheritDoc} */
    @Override
    public BuilderService getBuilderService() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ImageFactoryService getImageFactoryService() {
        return new DjlImageFactoryService();
    }

    /** {@inheritDoc} */
    @Override
    public ImplementationService getImplementationService() {
        return new DjlImplementationService();
    }
}
