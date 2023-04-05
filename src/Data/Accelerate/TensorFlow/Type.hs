{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}

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

type family Type64 a where
  Type64 Int = Int64
  Type64 Word = Word64
  Type64 a = a
  
type OneOf types a = TF.OneOf types (Type64 a)
data OneOfDict types a where
  OneOfDict :: (OneOf types a) => OneOfDict types a

type TensorType a = TF.TensorType (Type64 a)
data TensorTypeDict a where
  TensorTypeDict :: TensorType a => TensorTypeDict a

type VectorType a = (S.Storable a, TF.TensorDataType S.Vector (Type64 a), TF.TensorType (Type64 a))
data VectorTypeDict a where
  VectorTypeDict :: VectorType a => VectorTypeDict a

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
type TFOrd    = TFNum TF.\\ '[Complex Double, Complex Float]
type TFInt    = TFOrd TF.\\ '[Double, Float]

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