package ai.djl.jsr381.spi;

import javax.visrec.spi.ImplementationService;

/**
 * DJL' {@link ImplementationService}.
 *
 * @author Frank Liu
 */
public class DjlImplementationService extends ImplementationService {

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return "DJL";
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "0.3.0";
    }
}
