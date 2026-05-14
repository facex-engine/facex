/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef FRVT11_H_
#define FRVT11_H_

#include <cstdint>
#include <string>
#include <vector>

#include <frvt_structs.h>

namespace FRVT_11 {

/**
 * @brief
 * The interface to FRVT 1:1 implementation
 *
 * @details
 * The submission software under test will implement this interface by
 * sub-classing this class and implementing each method therein.
 */
class Interface {
public:
    virtual ~Interface() {}

    /**
     * @brief This function initializes the implementation under test.  It will
     * be called by the NIST application before any call to createTemplate() or
     * matchTemplates().  The implementation under test should set all parameters.
     * This function will be called N=1 times by the NIST application, prior to
     * parallelizing M >= 1 calls to createTemplate() via fork().
     *
     * @param[in] configDir
     * A read-only directory containing any developer-supplied configuration
     * parameters or run-time data files.  The name of this directory is
     * assigned by NIST, not hardwired by the provider.  The names of the
     * files in this directory are hardwired in the implementation and are
     * unrestricted.
     */
    virtual FRVT::ReturnStatus
    initialize(const std::string &configDir) = 0;

    /**
     * @brief This function supports template generation from one or more face images of 
     * exactly one person.  It takes a vector of images and outputs a proprietary template
     * and associated eye coordinates.  The vectors to store the template and
     * eye coordinates will be initially empty, and it is up to the implementation
     * to populate them with the appropriate data.  In all cases, even when unable
     * to extract features, the output shall be a template that may be passed to
     * the match_templates function without error.  That is, this routine must
     * internally encode "template creation failed" and the matcher must
     * transparently handle this.
     *
     * @param[in] faces
     * Implementations must alter their behavior according to the number of
     * images contained in the structure and the TemplateRole type.
     * @param[in] role
     * Label describing the type/role of the template to be generated
     * @param[out] templ
     * The output template.  The format is entirely unregulated.  This will be
     * an empty vector when passed into the function, and the implementation
     * can resize and populate it with the appropriate data.
     * @param[out] eyeCoordinates
     * For each input image in the vector, the function shall return the
     * estimated eye centers. This will be an empty vector when passed into the
     * function, and the implementation shall populate it with the appropriate
     * number of entries.  Values in eyeCoordinates[i] shall correspond to faces[i].
     */
    virtual FRVT::ReturnStatus
    createFaceTemplate(
        const std::vector<FRVT::Image> &faces,
        FRVT::TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<FRVT::EyePair> &eyeCoordinates) = 0;

    /**
     * @brief This function supports template generation from one or more iris images of
     * exactly one person.  It takes a vector of images and outputs a proprietary template
     * and associated iris locations.  The vectors to store the template and
     * eye coordinates will be initially empty, and it is up to the implementation
     * to populate them with the appropriate data.  In all cases, even when unable
     * to extract features, the output shall be a template that may be passed to
     * the match_templates function without error.  That is, this routine must
     * internally encode "template creation failed" and the matcher must
     * transparently handle this.
     *
     * @param[in] irises
     * Implementations must alter their behavior according to the number of
     * images contained in the structure and the TemplateRole type.
     * @param[in] role
     * Label describing the type/role of the template to be generated
     * @param[out] templ
     * The output template.  The format is entirely unregulated.  This will be
     * an empty vector when passed into the function, and the implementation
     * can resize and populate it with the appropriate data.
     * @param[out] irisLocations 
     * For each input image in the vector, the function shall return the
     * estimated iris locations. This will be an empty vector when passed into the
     * function, and the implementation shall populate it with the appropriate
     * number of entries.  Values in irisLocations[i] shall correspond to irises[i].
     */
    virtual FRVT::ReturnStatus
    createIrisTemplate(
        const std::vector<FRVT::Image> &irises,
        FRVT::TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<FRVT::IrisAnnulus> &irisLocations) = 0;

    /**
     * @brief This function supports template generation of one or more people detected 
     * from a single image.  It takes a single input image and outputs one or more proprietary 
     * templates and associated eye coordinates based on the number of people detected.  
     * The vectors to store the template(s) and eye coordinates will be initially empty, 
     * and it is up to the implementation to populate them with the appropriate data.  
     * If the implementation is unable to extract features, the output shall still contain a 
     * single template that may be passed to the match_templates function without error.  
     * That is, this routine must internally encode "template creation failed" and the 
     * matcher must transparently handle this.
     *
     * param[in] image
     * A single image that contains one or more people in the photo
     * param[in] role
     * Label describing the type/role of the template to be generated
     * param[out] templs
     * A vector of output template(s).  The format of the template(s) is entirely 
     * unregulated.  This will be an empty vector when passed into the function, and 
     * the implementation can resize and populate it with the appropriate data.
     * param[out] eyeCoordinates
     * For each person detected in the image, the function shall return the
     * estimated eye centers. This will be an empty vector when passed into the
     * function, and the implementation shall populate it with the appropriate
     * number of entries.  Values in eyeCoordinates[i] shall correspond to templs[i].
     */
    virtual FRVT::ReturnStatus
    createFaceTemplate(
        const FRVT::Image &image,
        FRVT::TemplateRole role,
        std::vector<std::vector<uint8_t>> &templs,
        std::vector<FRVT::EyePair> &eyeCoordinates) = 0;

    /**
     * @brief This function compares two proprietary templates and outputs a
     * measure of similarity or dissimilarity between the enrollment template and verification template.
     * - For face recognition, a similarity score - higher is more similar
     * - For iris recognition, a non-negative measure of dissimilarity (maybe a distance) - lower is more similar
     * When either or both of the input templates are the result of a failed
     * template generation, the score shall be -1 and the function
     * return value shall be VerifTemplateError.
     *
     * @param[in] verifTemplate
     * A verification template from createTemplate(role=Verification_11).
     * The underlying data can be accessed via verifTemplate.data().  The size,
     * in bytes, of the template could be retrieved as verifTemplate.size().
     * @param[in] enrollTemplate
     * An enrollment template from createTemplate(role=Enrollment_11).
     * The underlying data can be accessed via enrollTemplate.data().  The size,
     * in bytes, of the template could be retrieved as enrollTemplate.size().
     * @param[out] similarity
     * A score resulting from comparison of the templates,
     * on the range [0,DBL_MAX].
     *
     */
    virtual FRVT::ReturnStatus
    matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &score) = 0;

    /**
     * @brief
     * Factory method to return a managed pointer to the Interface object.
     * @details
     * This function is implemented by the submitted library and must return
     * a managed pointer to the Interface object.
     *
     * @note
     * A possible implementation might be:
     * return (std::make_shared<Implementation>());
     */
    static std::shared_ptr<Interface>
    getImplementation();
};

/*
 * API versioning
 *
 * NIST code will extern the version number symbols.
 * Participant shall compile them into their core library.
 */
#ifdef NIST_EXTERN_API_VERSION
/** API major version number. */
extern uint16_t API_MAJOR_VERSION;
/** API minor version number. */
extern uint16_t API_MINOR_VERSION;
#else /* NIST_EXTERN_API_VERSION */
/** API major version number. */
uint16_t API_MAJOR_VERSION{6};
/** API minor version number. */
uint16_t API_MINOR_VERSION{0};
#endif /* NIST_EXTERN_API_VERSION */
}

#endif /* FRVT11_H_ */
