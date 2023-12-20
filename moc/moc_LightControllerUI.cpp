/****************************************************************************
** Meta object code from reading C++ file 'LightControllerUI.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../sources/controller/LightControllerUI.h"
#include <QtCore/qmetatype.h>

#if __has_include(<QtCore/qtmochelpers.h>)
#include <QtCore/qtmochelpers.h>
#else
QT_BEGIN_MOC_NAMESPACE
#endif


#include <memory>

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LightControllerUI.h' doesn't include <QObject>."
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
struct qt_meta_stringdata_CLASSLightControllerENDCLASS_t {};
static constexpr auto qt_meta_stringdata_CLASSLightControllerENDCLASS = QtMocHelpers::stringData(
    "LightController",
    "addPointLight",
    "",
    "deletePointLight",
    "addDirectionalLight",
    "deleteDirectionalLight",
    "addSpotLight",
    "deleteSpotLight"
);
#else  // !QT_MOC_HAS_STRING_DATA
struct qt_meta_stringdata_CLASSLightControllerENDCLASS_t {
    uint offsetsAndSizes[16];
    char stringdata0[16];
    char stringdata1[14];
    char stringdata2[1];
    char stringdata3[17];
    char stringdata4[20];
    char stringdata5[23];
    char stringdata6[13];
    char stringdata7[16];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CLASSLightControllerENDCLASS_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CLASSLightControllerENDCLASS_t qt_meta_stringdata_CLASSLightControllerENDCLASS = {
    {
        QT_MOC_LITERAL(0, 15),  // "LightController"
        QT_MOC_LITERAL(16, 13),  // "addPointLight"
        QT_MOC_LITERAL(30, 0),  // ""
        QT_MOC_LITERAL(31, 16),  // "deletePointLight"
        QT_MOC_LITERAL(48, 19),  // "addDirectionalLight"
        QT_MOC_LITERAL(68, 22),  // "deleteDirectionalLight"
        QT_MOC_LITERAL(91, 12),  // "addSpotLight"
        QT_MOC_LITERAL(104, 15)   // "deleteSpotLight"
    },
    "LightController",
    "addPointLight",
    "",
    "deletePointLight",
    "addDirectionalLight",
    "deleteDirectionalLight",
    "addSpotLight",
    "deleteSpotLight"
};
#undef QT_MOC_LITERAL
#endif // !QT_MOC_HAS_STRING_DATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSLightControllerENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   50,    2, 0x09,    1 /* Protected */,
       3,    0,   51,    2, 0x09,    2 /* Protected */,
       4,    0,   52,    2, 0x09,    3 /* Protected */,
       5,    0,   53,    2, 0x09,    4 /* Protected */,
       6,    0,   54,    2, 0x09,    5 /* Protected */,
       7,    0,   55,    2, 0x09,    6 /* Protected */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject LightController::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_CLASSLightControllerENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSLightControllerENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSLightControllerENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<LightController, std::true_type>,
        // method 'addPointLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'deletePointLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'addDirectionalLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'deleteDirectionalLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'addSpotLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'deleteSpotLight'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void LightController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<LightController *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->addPointLight(); break;
        case 1: _t->deletePointLight(); break;
        case 2: _t->addDirectionalLight(); break;
        case 3: _t->deleteDirectionalLight(); break;
        case 4: _t->addSpotLight(); break;
        case 5: _t->deleteSpotLight(); break;
        default: ;
        }
    }
    (void)_a;
}

const QMetaObject *LightController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LightController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSLightControllerENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int LightController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 6;
    }
    return _id;
}
QT_WARNING_POP
