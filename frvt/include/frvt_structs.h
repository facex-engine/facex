/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef FRVT_STRUCTS_H_
#define FRVT_STRUCTS_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace FRVT {
/**
 * @brief
 * Struct representing a single image
 */
typedef struct Image {
    /** Labels describing the type of image */
    enum class ImageDescription {
        /** Face image with unknown or unassigned collection conditions */
        FaceUnknown = 0,
        /** Face image, Frontal closely ISO/IEC 19794-5:2005 compliant. */
        FaceIso = 1,
        /** Face image from law enforcement booking processes, nominally frontal. */
        FaceMugshot = 2,
        /** Face image that might appear in a news source or magazine. The images are typically
         * typically well exposed and focused but exhibit pose and illumination variations. */
        FacePhotojournalism = 3,
        /** Unconstrained face, taken by amateur photographer, widely varying pose, illumination, resolution. */
        FaceWild = 4,
        /** Iris image with unknown or unassigned collection conditions */
        IrisUnknown = 5,
        /** Image of one iris from iris camera that illuminates the iris in NIR */
        IrisNIR = 6,
        /** Image of one iris from non-iris camera, with visible ambient illumination */
        IrisWild = 7
    };

    enum class Illuminant {
        /** Not specified */
        Unspecified = 0,
        /** Conventional visible light */
        Visible = 1,
        /** Near infrared, as used in conventional iris cameras and some face cameras e.g. outdoors */
        NIR = 2,
        /** Short wave infrared */
        SWIR = 3,
        /** Medium wave infrared */
        MWIR = 4,
        /** Long wave infrared - emissive */
        LWIR = 5
    };

    /** Labels describing whether it's the left or right iris */ 
    enum class IrisLR
    {
        /** Not specified */
        Unspecified = 0,
        /** Right iris */
        RightIris = 1,
        /** Left iris */
        LeftIris = 2
    };

    /** Number of pixels horizontally */
    uint16_t width;
    /** Number of pixels vertically */
    uint16_t height;
    /** Number of bits per pixel. Legal values are 8 and 24. */
    uint8_t depth;
    /** Managed pointer to raster scanned data.
     * Either RGB color or intensity.
     * If image_depth == 24 this points to  3WH bytes  RGBRGBRGB...
     * If image_depth ==  8 this points to  WH bytes  IIIIIII */
    std::shared_ptr<uint8_t> data;
    /** Single description of the image.  */     
    ImageDescription description;
    /** Source of light used to acquire the image */
    Illuminant illuminant;
    /** The iris label (left, right, or unspecified).  Not applicable for face images. */
    IrisLR irisLR;

    Image() :
        width{0},
        height{0},
        depth{24},
        description{ImageDescription::FaceUnknown},
        illuminant{Illuminant::Unspecified},
        irisLR{IrisLR::Unspecified}
        {}

    Image(
        uint16_t width,
        uint16_t height,
        uint8_t depth,
        std::shared_ptr<uint8_t> &data,
        ImageDescription description,
        Illuminant illuminant,
        IrisLR irisLR = IrisLR::Unspecified
        ) :
        width{width},
        height{height},
        depth{depth},
        data{data},
        description{description},
        illuminant{illuminant},
        irisLR{irisLR}
        {}

    /** @brief This function returns the size of the image data. */
    size_t
    size() const { return (width * height * (depth / 8)); }
} Image;

/**
 * @brief
 * Struct representing a piece of media
 */
typedef struct Media {
    /** Labels describing the type of media */
    enum class Label {
        /** Still photos of an individual */
        Image = 0,
        /** Sequential video frames of an individual */
        Video = 1
    };

    /** Type of media */
    Label type;
    /** Vector of still image(s) or video frames */
    std::vector<FRVT::Image> data;
    /** For video data, the frame rate in frames per second */
    uint8_t fps;

    Media() :
        type{Media::Label::Image},
        fps{0}
        {}

    Media(
        const Media::Label type,
        const std::vector<FRVT::Image> &data,
        const uint8_t fps
        ) :
        type{type},
        data{data},
        fps{fps}
        {}
} Media;

/** @brief 
 * Structure specifying the approximate horizontal center of the limbus 
 * in pixels of the iris in an image. Provides an estimate of the limbus 
 * center (limbusCenterX, limbusCenterY) and pupil (pupilRadius) and 
 * limbus (limbusRadius) radii. When provided, the estimates should be 
 * accurate to within a few pixels.
 */
typedef struct IrisAnnulus
{
    /** @brief X-coordinate of the limbus center. */
    uint16_t limbusCenterX;

    /** @brief Y-coordinate of the limbus center. */
    uint16_t limbusCenterY;

    /** @brief Estimate of pupil radius in pixels. */
    uint16_t pupilRadius;

    /** @brief Estimate of limbus radius in pixels. */
    uint16_t limbusRadius;

    IrisAnnulus() :
        limbusCenterX{0},
        limbusCenterY{0},
        pupilRadius{0},
        limbusRadius{0}
        {}
       
    IrisAnnulus(
        uint16_t limbusCenterX,
        uint16_t limbusCenterY,
        uint16_t pupilRadius,
        uint16_t limbusRadius
        ) :
        limbusCenterX{limbusCenterX},
        limbusCenterY{limbusCenterY},
        pupilRadius{pupilRadius},
        limbusRadius{limbusRadius}
        {} 
} IrisAnnulus;

/** Labels describing the type/role of the template
 * to be generated (provided as input to template generation)
 */
enum class TemplateRole {
    /** 1:1 enrollment template */
    Enrollment_11 = 0,
    /** 1:1 verification template */
    Verification_11 = 1,
    /** 1:N enrollment template */
    Enrollment_1N = 2,
    /** 1:N identification template */
    Search_1N = 3
};

/**
 * @brief
 * Return codes for functions specified in this API
 */
enum class ReturnCode {
    /** Success */
    Success = 0,
    /** Catch-all error */
    UnknownError = 1,
    /** Error reading configuration files */
    ConfigError = 2,
    /** Elective refusal to process the input */
    RefuseInput = 3,
    /** Involuntary failure to process the image */
    ExtractError = 4,
    /** Cannot parse the input data */
    ParseError = 5,
    /** Elective refusal to produce a template */
    TemplateCreationError = 6,
    /** Either or both of the input templates were result of failed feature extraction */
    VerifTemplateError = 7,
    /** Unable to detect a face in the image */
    FaceDetectionError = 8,
    /** The implementation cannot support the number of input images */
    NumDataError = 9,
    /** Template file is an incorrect format or defective */
    TemplateFormatError = 10,
    /** An operation on the enrollment directory failed (e.g. permission, space) */
    EnrollDirError = 11,
    /** Cannot locate the input data - the input files or names seem incorrect */
    InputLocationError = 12,
    /** Memory allocation failed (e.g. out of memory) */
    MemoryError = 13,
    /** Error occurred during the 1:1 match operation */
    MatchError = 14,
    /** Failure to generate a quality score on the input image */
    QualityAssessmentError = 15,
    /** Function is not implemented */
    NotImplemented = 16,
    /** Vendor-defined failure */
    VendorError = 17
};

/** Output stream operator for a ReturnCode object. */
inline std::ostream&
operator<<(
    std::ostream &s,
    const ReturnCode &rc)
{
    switch (rc) {
    case ReturnCode::Success:
        return (s << "Success");
    case ReturnCode::UnknownError:
        return (s << "Unknown Error");
    case ReturnCode::ConfigError:
        return (s << "Error reading configuration files");
    case ReturnCode::RefuseInput:
        return (s << "Elective refusal to process the input");
    case ReturnCode::ExtractError:
        return (s << "Involuntary failure to process the image");
    case ReturnCode::ParseError:
        return (s << "Cannot parse the input data");
    case ReturnCode::TemplateCreationError:
        return (s << "Elective refusal to produce a template");
    case ReturnCode::VerifTemplateError:
        return (s << "Either or both of the input templates were result of "
                "failed feature extraction");
    case ReturnCode::FaceDetectionError:
        return (s << "Unable to detect a face in the image");
    case ReturnCode::NumDataError:
        return (s << "Number of input images not supported");
    case ReturnCode::TemplateFormatError:
        return (s << "Template file is an incorrect format or defective");
    case ReturnCode::EnrollDirError:
        return (s << "An operation on the enrollment directory failed");
    case ReturnCode::InputLocationError:
        return (s << "Cannot locate the input data - the input files or names "
                "seem incorrect");
    case ReturnCode::MemoryError:
        return (s << "Memory allocation failed (e.g. out of memory)");
    case ReturnCode::MatchError:
        return (s << "Error occurred during the 1:1 match operation");
    case ReturnCode::QualityAssessmentError:
        return (s << "Failure to generate a quality score on the input image");
    case ReturnCode::NotImplemented:
        return (s << "Function is not implemented");
    case ReturnCode::VendorError:
        return (s << "Vendor-defined error");
    default:
        return (s << "Undefined error");
    }
}

/**
 * @brief
 * A structure to contain information about a failure by the software
 * under test.
 *
 * @details
 * An object of this class allows the software to return some information
 * from a function call. The string within this object can be optionally
 * set to provide more information for debugging etc. The status code
 * will be set by the function to Success on success, or one of the
 * other codes on failure.
 */
typedef struct ReturnStatus {
    /** @brief Return status code */
    ReturnCode code;
    /** @brief Optional information string */
    std::string info;

    ReturnStatus() :
        code{ReturnCode::UnknownError},
        info{""}
        {}
    /**
     * @brief
     * Create a ReturnStatus object.
     *
     * @param[in] code
     * The return status code; required.
     * @param[in] info
     * The optional information string.
     */
    ReturnStatus(
        const ReturnCode code,
        const std::string &info = ""
        ) :
        code{code},
        info{info}
        {}
} ReturnStatus;

typedef struct EyePair
{
    /** If the left eye coordinates have been computed and
     * assigned successfully, this value should be set to true,
     * otherwise false. */
    bool isLeftAssigned;
    /** If the right eye coordinates have been computed and
     * assigned successfully, this value should be set to true,
     * otherwise false. */
    bool isRightAssigned;
    /** X and Y coordinate of the midpoint between the two canthi of the subject's left eye.  If the
     * eye coordinate is out of range (e.g. x < 0 or x >= width), isLeftAssigned
     * should be set to false, and the left eye coordinates will be ignored. */
    uint16_t xleft;
    uint16_t yleft;
    /** X and Y coordinate of the midpoint between the two canthi of the subject's right eye.  If the
     * eye coordinate is out of range (e.g. x < 0 or x >= width), isRightAssigned
     * should be set to false, and the right eye coordinates will be ignored. */
    uint16_t xright;
    uint16_t yright;

    EyePair() :
        isLeftAssigned{false},
        isRightAssigned{false},
        xleft{0},
        yleft{0},
        xright{0},
        yright{0}
        {}

    EyePair(
        bool isLeftAssigned,
        bool isRightAssigned,
        uint16_t xleft,
        uint16_t yleft,
        uint16_t xright,
        uint16_t yright
        ) :
        isLeftAssigned{isLeftAssigned},
        isRightAssigned{isRightAssigned},
        xleft{xleft},
        yleft{yleft},
        xright{xright},
        yright{yright}
        {}
} EyePair;

/*
* Versioning
*
* NIST code will extern the version number symbols. Participant
* shall compile them into their core library.
*/
#ifdef NIST_EXTERN_FRVT_STRUCTS_VERSION
/** major version number. */
extern uint16_t FRVT_STRUCTS_MAJOR_VERSION;
/** minor version number. */
extern uint16_t FRVT_STRUCTS_MINOR_VERSION;
#else /* NIST_EXTERN_FRVT_STRUCTS_VERSION */
/** major version number. */
uint16_t FRVT_STRUCTS_MAJOR_VERSION{3};
/** minor version number. */
uint16_t FRVT_STRUCTS_MINOR_VERSION{1};
#endif /* NIST_EXTERN_FRVT_STRUCTS_VERSION */
}

#endif /* FRVT_STRUCTS_H_ */
