{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}

module Data.Accelerate.TensorFlow.Type where
import Data.Int ( Int8, Int16, Int32, Int64 )
import Data.Word ( Word8, Word16, Word32, Word64 )
import qualified TensorFlow.Types as TF
import qualified Data.Vector.Storable as S
import Data.Complex ( Complex )
import Data.Array.Accelerate.Type
    ( FloatingType(..),
      IntegralType(..),
      NumType(..),
      SingleType(..), ScalarType(..) )
import Data.Array.Accelerate.Representation.Type (TypeR, TupR (..))

type family Type64 a where
  Type64 Int   = Int64
  Type64 Word  = Word64
  Type64 a     = a

type OneOf types a = TF.OneOf types (Type64 a)
data OneOfDict types a where
  OneOfDict :: (OneOf types a) => OneOfDict types a

type TensorType a = TF.TensorType (Type64 a)
data TensorTypeDict a where
  TensorTypeDict :: TensorType a => TensorTypeDict a

type VectorType a = (S.Storable a, TF.TensorDataType S.Vector a, TF.TensorType a)
data VectorTypeDict a where
  VectorTypeDict :: VectorType (Type64 a) => VectorTypeDict a

toType64 :: ScalarType e -> ScalarType (Type64 e)
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = SingleScalarType (NumSingleType (IntegralNumType TypeInt64))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = SingleScalarType (NumSingleType (IntegralNumType TypeInt8))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = SingleScalarType (NumSingleType (IntegralNumType TypeInt16))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = SingleScalarType (NumSingleType (IntegralNumType TypeInt32))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = SingleScalarType (NumSingleType (IntegralNumType TypeInt64))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = SingleScalarType (NumSingleType (IntegralNumType TypeWord64))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = SingleScalarType (NumSingleType (IntegralNumType TypeWord8))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = SingleScalarType (NumSingleType (IntegralNumType TypeWord16))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = SingleScalarType (NumSingleType (IntegralNumType TypeWord32))
toType64 (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = SingleScalarType (NumSingleType (IntegralNumType TypeWord64))
toType64 (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = SingleScalarType (NumSingleType (FloatingNumType TypeHalf))
toType64 (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = SingleScalarType (NumSingleType (FloatingNumType TypeFloat))
toType64 (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = SingleScalarType (NumSingleType (FloatingNumType TypeDouble))
toType64 st@(VectorScalarType _)                                         = st

toType64' :: ScalarType e -> e -> Type64 e
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    t = fromIntegral t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   t = fromIntegral t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) t = t
toType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) t = t
toType64' (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   t = t
toType64' (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  t = t
toType64' (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) t = t
toType64' (VectorScalarType _) t = t

fromType64' :: ScalarType e -> Type64 e -> e
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    t = fromIntegral t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   t = fromIntegral t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) t = t
fromType64' (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) t = t
fromType64' (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   t = t
fromType64' (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  t = t
fromType64' (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) t = t
fromType64' (VectorScalarType _) t = t

type TFAll = '[Complex Double,
               Complex Float,
               Bool,
               Int16,
               Int32,
               Int64,
               Int8,
               Word16,
               Word32,
               Word64,
               Word8,
               Double,
               Float]

type TFNum    = TFAll TF.\\ '[Bool]
type TFNum'   = TFNum TF.\\ '[Word8]
type TFOrd    = TFNum TF.\\ '[Complex Double, Complex Float]
type TFInt    = TFOrd TF.\\ '[Double, Float]
type TFInt'   = TFInt TF.\\ '[Word8]
type TFMod    = '[Int32, Int64, Word16, Double, Float]

type TFFloat = '[Double, Float]
tfFloatDict :: FloatingType a -> OneOfDict TFFloat a
tfFloatDict TypeFloat  = OneOfDict
tfFloatDict TypeDouble = OneOfDict 
tfFloatDict TypeHalf   = error "not a TF float type"

tfNumDict :: NumType a -> OneOfDict TFNum a
tfNumDict (IntegralNumType TypeInt)    = OneOfDict
tfNumDict (IntegralNumType TypeInt8)   = OneOfDict
tfNumDict (IntegralNumType TypeInt16)  = OneOfDict
tfNumDict (IntegralNumType TypeInt32)  = OneOfDict
tfNumDict (IntegralNumType TypeInt64)  = OneOfDict
tfNumDict (IntegralNumType TypeWord)   = OneOfDict
tfNumDict (IntegralNumType TypeWord8)  = OneOfDict
tfNumDict (IntegralNumType TypeWord16) = OneOfDict
tfNumDict (IntegralNumType TypeWord32) = OneOfDict
tfNumDict (IntegralNumType TypeWord64) = OneOfDict
tfNumDict (FloatingNumType TypeHalf)   = error "not a TF num type"
tfNumDict (FloatingNumType TypeFloat)  = OneOfDict
tfNumDict (FloatingNumType TypeDouble) = OneOfDict

tfModDict :: IntegralType a -> OneOfDict TFMod a
tfModDict TypeInt    = OneOfDict
tfModDict TypeInt8   = error "not a TF mod type"
tfModDict TypeInt16  = error "not a TF mod type"
tfModDict TypeInt32  = OneOfDict
tfModDict TypeInt64  = OneOfDict
tfModDict TypeWord   = OneOfDict
tfModDict TypeWord8  = error "not a TF mod type"
tfModDict TypeWord16 = OneOfDict
tfModDict TypeWord32 = OneOfDict
tfModDict TypeWord64 = OneOfDict

tfNum'Dict :: NumType a -> OneOfDict TFNum' a
tfNum'Dict (IntegralNumType TypeInt)    = OneOfDict
tfNum'Dict (IntegralNumType TypeInt8)   = OneOfDict
tfNum'Dict (IntegralNumType TypeInt16)  = OneOfDict
tfNum'Dict (IntegralNumType TypeInt32)  = OneOfDict
tfNum'Dict (IntegralNumType TypeInt64)  = OneOfDict
tfNum'Dict (IntegralNumType TypeWord)   = OneOfDict
tfNum'Dict (IntegralNumType TypeWord8)  = error "not a TF num type"
tfNum'Dict (IntegralNumType TypeWord16) = OneOfDict
tfNum'Dict (IntegralNumType TypeWord32) = OneOfDict
tfNum'Dict (IntegralNumType TypeWord64) = OneOfDict
tfNum'Dict (FloatingNumType TypeHalf)   = error "not a TF num type"
tfNum'Dict (FloatingNumType TypeFloat)  = OneOfDict
tfNum'Dict (FloatingNumType TypeDouble) = OneOfDict

tfOrdDict :: SingleType a -> OneOfDict TFOrd a
tfOrdDict (NumSingleType (IntegralNumType TypeInt))    = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeInt8))   = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeInt16))  = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeInt32))  = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeInt64))  = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeWord))   = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeWord8))  = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeWord16)) = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeWord32)) = OneOfDict
tfOrdDict (NumSingleType (IntegralNumType TypeWord64)) = OneOfDict
tfOrdDict (NumSingleType (FloatingNumType TypeHalf))   = error "not a TF ord type"
tfOrdDict (NumSingleType (FloatingNumType TypeFloat))  = OneOfDict
tfOrdDict (NumSingleType (FloatingNumType TypeDouble)) = OneOfDict

tfIntDict :: IntegralType a -> OneOfDict TFInt a
tfIntDict TypeInt    = OneOfDict
tfIntDict TypeInt8   = OneOfDict
tfIntDict TypeInt16  = OneOfDict
tfIntDict TypeInt32  = OneOfDict
tfIntDict TypeInt64  = OneOfDict
tfIntDict TypeWord   = OneOfDict
tfIntDict TypeWord8  = OneOfDict
tfIntDict TypeWord16 = OneOfDict
tfIntDict TypeWord32 = OneOfDict
tfIntDict TypeWord64 = OneOfDict

tfTensorTypeDict' :: TypeR a -> TensorTypeDict a
tfTensorTypeDict' (TupRsingle s) = tfTensorTypeDict s
tfTensorTypeDict' _ = error "not a tf tensortype"

tfTensorTypeDict :: ScalarType a -> TensorTypeDict a
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a TF tensor type"
tfTensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = TensorTypeDict
tfTensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = TensorTypeDict
tfTensorTypeDict (VectorScalarType _) = error "not a TF tensor type"

tfVectorTypeDict :: ScalarType a -> VectorTypeDict a
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = error "not a TF vector type"
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = error "not a TF vector type"
tfVectorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = error "not a TF vector type"
tfVectorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a TF vector type"
tfVectorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = VectorTypeDict
tfVectorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = VectorTypeDict
tfVectorTypeDict (VectorScalarType _) = error "not a TF vector type"

tfAllDict' :: TypeR a -> OneOfDict TFAll a
tfAllDict' (TupRsingle s) = tfAllDict s
tfAllDict' _ = error "not a tf all type"

tfAllDict :: ScalarType a -> OneOfDict TFAll a
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a TF all type"
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = OneOfDict
tfAllDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = OneOfDict
tfAllDict (VectorScalarType _) = error "not a TF all type"

zero :: ScalarType t -> t
zero (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = 0
zero (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = 0
zero (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = 0
zero (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = 0
zero (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = 0
zero (VectorScalarType _)                                            = error "not a zero type"

one :: ScalarType t -> t
one (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = 1
one (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = 1
one (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = 1
one (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = 1
one (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = 1
one (VectorScalarType _)                                            = error "not a one type"