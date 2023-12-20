/****************************************************************************
** Meta object code from reading C++ file 'GUIWindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../sources/controller/GUIWindow.h"
#include <QtCore/qmetatype.h>

#if __has_include(<QtCore/qtmochelpers.h>)
#include <QtCore/qtmochelpers.h>
#else
QT_BEGIN_MOC_NAMESPACE
#endif


#include <memory>

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GUIWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS_t {};
static constexpr auto qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS = QtMocHelpers::stringData(
    "axomae::Controller",
    "import_image",
    "",
    "import_3DOBJ",
    "open_project",
    "save_project",
    "save_image",
    "greyscale_average",
    "greyscale_luminance",
    "use_scharr",
    "use_prewitt",
    "use_sobel",
    "use_gpgpu",
    "checked",
    "use_object_space",
    "use_tangent_space",
    "change_nmap_factor",
    "factor",
    "change_nmap_attenuation",
    "atten",
    "compute_dudv",
    "change_dudv_nmap",
    "compute_projection",
    "next_mesh",
    "previous_mesh",
    "project_uv_normals",
    "smooth_edge",
    "sharpen_edge",
    "undo",
    "redo",
    "set_renderer_gamma_value",
    "gamma",
    "set_renderer_exposure_value",
    "exposure",
    "reset_renderer_camera",
    "set_renderer_no_post_process",
    "set_renderer_edge_post_process",
    "set_renderer_sharpen_post_process",
    "set_renderer_blurr_post_process",
    "set_rasterizer_point",
    "set_rasterizer_fill",
    "set_rasterizer_wireframe",
    "set_display_boundingbox",
    "display",
    "update_smooth_factor"
);
#else  // !QT_MOC_HAS_STRING_DATA
struct qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS_t {
    uint offsetsAndSizes[90];
    char stringdata0[19];
    char stringdata1[13];
    char stringdata2[1];
    char stringdata3[13];
    char stringdata4[13];
    char stringdata5[13];
    char stringdata6[11];
    char stringdata7[18];
    char stringdata8[20];
    char stringdata9[11];
    char stringdata10[12];
    char stringdata11[10];
    char stringdata12[10];
    char stringdata13[8];
    char stringdata14[17];
    char stringdata15[18];
    char stringdata16[19];
    char stringdata17[7];
    char stringdata18[24];
    char stringdata19[6];
    char stringdata20[13];
    char stringdata21[17];
    char stringdata22[19];
    char stringdata23[10];
    char stringdata24[14];
    char stringdata25[19];
    char stringdata26[12];
    char stringdata27[13];
    char stringdata28[5];
    char stringdata29[5];
    char stringdata30[25];
    char stringdata31[6];
    char stringdata32[28];
    char stringdata33[9];
    char stringdata34[22];
    char stringdata35[29];
    char stringdata36[31];
    char stringdata37[34];
    char stringdata38[32];
    char stringdata39[21];
    char stringdata40[20];
    char stringdata41[25];
    char stringdata42[24];
    char stringdata43[8];
    char stringdata44[21];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS_t qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS = {
    {
        QT_MOC_LITERAL(0, 18),  // "axomae::Controller"
        QT_MOC_LITERAL(19, 12),  // "import_image"
        QT_MOC_LITERAL(32, 0),  // ""
        QT_MOC_LITERAL(33, 12),  // "import_3DOBJ"
        QT_MOC_LITERAL(46, 12),  // "open_project"
        QT_MOC_LITERAL(59, 12),  // "save_project"
        QT_MOC_LITERAL(72, 10),  // "save_image"
        QT_MOC_LITERAL(83, 17),  // "greyscale_average"
        QT_MOC_LITERAL(101, 19),  // "greyscale_luminance"
        QT_MOC_LITERAL(121, 10),  // "use_scharr"
        QT_MOC_LITERAL(132, 11),  // "use_prewitt"
        QT_MOC_LITERAL(144, 9),  // "use_sobel"
        QT_MOC_LITERAL(154, 9),  // "use_gpgpu"
        QT_MOC_LITERAL(164, 7),  // "checked"
        QT_MOC_LITERAL(172, 16),  // "use_object_space"
        QT_MOC_LITERAL(189, 17),  // "use_tangent_space"
        QT_MOC_LITERAL(207, 18),  // "change_nmap_factor"
        QT_MOC_LITERAL(226, 6),  // "factor"
        QT_MOC_LITERAL(233, 23),  // "change_nmap_attenuation"
        QT_MOC_LITERAL(257, 5),  // "atten"
        QT_MOC_LITERAL(263, 12),  // "compute_dudv"
        QT_MOC_LITERAL(276, 16),  // "change_dudv_nmap"
        QT_MOC_LITERAL(293, 18),  // "compute_projection"
        QT_MOC_LITERAL(312, 9),  // "next_mesh"
        QT_MOC_LITERAL(322, 13),  // "previous_mesh"
        QT_MOC_LITERAL(336, 18),  // "project_uv_normals"
        QT_MOC_LITERAL(355, 11),  // "smooth_edge"
        QT_MOC_LITERAL(367, 12),  // "sharpen_edge"
        QT_MOC_LITERAL(380, 4),  // "undo"
        QT_MOC_LITERAL(385, 4),  // "redo"
        QT_MOC_LITERAL(390, 24),  // "set_renderer_gamma_value"
        QT_MOC_LITERAL(415, 5),  // "gamma"
        QT_MOC_LITERAL(421, 27),  // "set_renderer_exposure_value"
        QT_MOC_LITERAL(449, 8),  // "exposure"
        QT_MOC_LITERAL(458, 21),  // "reset_renderer_camera"
        QT_MOC_LITERAL(480, 28),  // "set_renderer_no_post_process"
        QT_MOC_LITERAL(509, 30),  // "set_renderer_edge_post_process"
        QT_MOC_LITERAL(540, 33),  // "set_renderer_sharpen_post_pro..."
        QT_MOC_LITERAL(574, 31),  // "set_renderer_blurr_post_process"
        QT_MOC_LITERAL(606, 20),  // "set_rasterizer_point"
        QT_MOC_LITERAL(627, 19),  // "set_rasterizer_fill"
        QT_MOC_LITERAL(647, 24),  // "set_rasterizer_wireframe"
        QT_MOC_LITERAL(672, 23),  // "set_display_boundingbox"
        QT_MOC_LITERAL(696, 7),  // "display"
        QT_MOC_LITERAL(704, 20)   // "update_smooth_factor"
    },
    "axomae::Controller",
    "import_image",
    "",
    "import_3DOBJ",
    "open_project",
    "save_project",
    "save_image",
    "greyscale_average",
    "greyscale_luminance",
    "use_scharr",
    "use_prewitt",
    "use_sobel",
    "use_gpgpu",
    "checked",
    "use_object_space",
    "use_tangent_space",
    "change_nmap_factor",
    "factor",
    "change_nmap_attenuation",
    "atten",
    "compute_dudv",
    "change_dudv_nmap",
    "compute_projection",
    "next_mesh",
    "previous_mesh",
    "project_uv_normals",
    "smooth_edge",
    "sharpen_edge",
    "undo",
    "redo",
    "set_renderer_gamma_value",
    "gamma",
    "set_renderer_exposure_value",
    "exposure",
    "reset_renderer_camera",
    "set_renderer_no_post_process",
    "set_renderer_edge_post_process",
    "set_renderer_sharpen_post_process",
    "set_renderer_blurr_post_process",
    "set_rasterizer_point",
    "set_rasterizer_fill",
    "set_rasterizer_wireframe",
    "set_display_boundingbox",
    "display",
    "update_smooth_factor"
};
#undef QT_MOC_LITERAL
#endif // !QT_MOC_HAS_STRING_DATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSaxomaeSCOPEControllerENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
      37,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,  236,    2, 0x0a,    1 /* Public */,
       3,    0,  237,    2, 0x0a,    2 /* Public */,
       4,    0,  238,    2, 0x0a,    3 /* Public */,
       5,    0,  239,    2, 0x0a,    4 /* Public */,
       6,    0,  240,    2, 0x0a,    5 /* Public */,
       7,    0,  241,    2, 0x0a,    6 /* Public */,
       8,    0,  242,    2, 0x0a,    7 /* Public */,
       9,    0,  243,    2, 0x0a,    8 /* Public */,
      10,    0,  244,    2, 0x0a,    9 /* Public */,
      11,    0,  245,    2, 0x0a,   10 /* Public */,
      12,    1,  246,    2, 0x0a,   11 /* Public */,
      14,    0,  249,    2, 0x0a,   13 /* Public */,
      15,    0,  250,    2, 0x0a,   14 /* Public */,
      16,    1,  251,    2, 0x0a,   15 /* Public */,
      18,    1,  254,    2, 0x0a,   17 /* Public */,
      20,    0,  257,    2, 0x0a,   19 /* Public */,
      21,    1,  258,    2, 0x0a,   20 /* Public */,
      22,    0,  261,    2, 0x0a,   22 /* Public */,
      23,    0,  262,    2, 0x0a,   23 /* Public */,
      24,    0,  263,    2, 0x0a,   24 /* Public */,
      25,    0,  264,    2, 0x0a,   25 /* Public */,
      26,    0,  265,    2, 0x0a,   26 /* Public */,
      27,    0,  266,    2, 0x0a,   27 /* Public */,
      28,    0,  267,    2, 0x0a,   28 /* Public */,
      29,    0,  268,    2, 0x0a,   29 /* Public */,
      30,    1,  269,    2, 0x0a,   30 /* Public */,
      32,    1,  272,    2, 0x0a,   32 /* Public */,
      34,    0,  275,    2, 0x0a,   34 /* Public */,
      35,    0,  276,    2, 0x0a,   35 /* Public */,
      36,    0,  277,    2, 0x0a,   36 /* Public */,
      37,    0,  278,    2, 0x0a,   37 /* Public */,
      38,    0,  279,    2, 0x0a,   38 /* Public */,
      39,    0,  280,    2, 0x0a,   39 /* Public */,
      40,    0,  281,    2, 0x0a,   40 /* Public */,
      41,    0,  282,    2, 0x0a,   41 /* Public */,
      42,    1,  283,    2, 0x0a,   42 /* Public */,
      44,    1,  286,    2, 0x09,   44 /* Protected */,

 // slots: parameters
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Bool,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   13,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Int,   19,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   31,
    QMetaType::Void, QMetaType::Int,   33,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   43,
    QMetaType::Void, QMetaType::Int,   17,

       0        // eod
};

Q_CONSTINIT const QMetaObject axomae::Controller::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSaxomaeSCOPEControllerENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<Controller, std::true_type>,
        // method 'import_image'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'import_3DOBJ'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'open_project'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'save_project'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'save_image'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'greyscale_average'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'greyscale_luminance'
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'use_scharr'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'use_prewitt'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'use_sobel'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'use_gpgpu'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'use_object_space'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'use_tangent_space'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'change_nmap_factor'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'change_nmap_attenuation'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'compute_dudv'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'change_dudv_nmap'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'compute_projection'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'next_mesh'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'previous_mesh'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'project_uv_normals'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'smooth_edge'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'sharpen_edge'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'undo'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'redo'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_renderer_gamma_value'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'set_renderer_exposure_value'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'reset_renderer_camera'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_renderer_no_post_process'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_renderer_edge_post_process'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_renderer_sharpen_post_process'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_renderer_blurr_post_process'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_rasterizer_point'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_rasterizer_fill'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_rasterizer_wireframe'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'set_display_boundingbox'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'update_smooth_factor'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>
    >,
    nullptr
} };

void axomae::Controller::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Controller *>(_o);
        (void)_t;
        switch (_id) {
        case 0: { bool _r = _t->import_image();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 1: { bool _r = _t->import_3DOBJ();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 2: { bool _r = _t->open_project();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 3: { bool _r = _t->save_project();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 4: { bool _r = _t->save_image();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 5: { bool _r = _t->greyscale_average();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 6: { bool _r = _t->greyscale_luminance();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 7: _t->use_scharr(); break;
        case 8: _t->use_prewitt(); break;
        case 9: _t->use_sobel(); break;
        case 10: _t->use_gpgpu((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 11: _t->use_object_space(); break;
        case 12: _t->use_tangent_space(); break;
        case 13: _t->change_nmap_factor((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 14: _t->change_nmap_attenuation((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 15: _t->compute_dudv(); break;
        case 16: _t->change_dudv_nmap((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 17: _t->compute_projection(); break;
        case 18: _t->next_mesh(); break;
        case 19: _t->previous_mesh(); break;
        case 20: _t->project_uv_normals(); break;
        case 21: _t->smooth_edge(); break;
        case 22: _t->sharpen_edge(); break;
        case 23: _t->undo(); break;
        case 24: _t->redo(); break;
        case 25: _t->set_renderer_gamma_value((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 26: _t->set_renderer_exposure_value((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 27: _t->reset_renderer_camera(); break;
        case 28: _t->set_renderer_no_post_process(); break;
        case 29: _t->set_renderer_edge_post_process(); break;
        case 30: _t->set_renderer_sharpen_post_process(); break;
        case 31: _t->set_renderer_blurr_post_process(); break;
        case 32: _t->set_rasterizer_point(); break;
        case 33: _t->set_rasterizer_fill(); break;
        case 34: _t->set_rasterizer_wireframe(); break;
        case 35: _t->set_display_boundingbox((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 36: _t->update_smooth_factor((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject *axomae::Controller::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *axomae::Controller::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSaxomaeSCOPEControllerENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int axomae::Controller::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 37)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 37;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 37)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 37;
    }
    return _id;
}
QT_WARNING_POP
